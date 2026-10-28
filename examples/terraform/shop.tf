# Example AWS web shop: Route 53 -> CloudFront (+WAF) -> ALB -> Auto Scaling group,
# with ElastiCache, RDS, SQS + a Lambda worker, S3, and CloudWatch alarms.
# Import it in the app with "Import IaC" to see it in 3D. It is not meant to be applied as is.

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "private" {
  vpc_id     = aws_vpc.main.id
  cidr_block = "10.0.1.0/24"
}

resource "aws_route53_zone" "shop" {
  name = "shop.example.com"
}

resource "aws_route53_record" "www" {
  zone_id = aws_route53_zone.shop.zone_id
  name    = "www.shop.example.com"
  type    = "A"
  alias {
    name                   = aws_cloudfront_distribution.cdn.domain_name
    zone_id                = aws_cloudfront_distribution.cdn.hosted_zone_id
    evaluate_target_health = false
  }
}

resource "aws_wafv2_web_acl" "edge" {
  name  = "shop-edge"
  scope = "CLOUDFRONT"
  default_action {
    allow {}
  }
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "shop-edge"
    sampled_requests_enabled   = true
  }
}

resource "aws_cloudfront_distribution" "cdn" {
  enabled    = true
  web_acl_id = aws_wafv2_web_acl.edge.arn
  origin {
    domain_name = aws_lb.web.dns_name
    origin_id   = "alb"
  }
  origin {
    domain_name = aws_s3_bucket.assets.bucket_regional_domain_name
    origin_id   = "assets"
  }
}

resource "aws_security_group" "alb" {
  name   = "alb"
  vpc_id = aws_vpc.main.id
}

resource "aws_security_group" "app" {
  name   = "app"
  vpc_id = aws_vpc.main.id
  ingress {
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
}

resource "aws_security_group" "data" {
  name   = "data"
  vpc_id = aws_vpc.main.id
  ingress {
    from_port       = 5432
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }
}

resource "aws_lb" "web" {
  name            = "shop-web"
  security_groups = [aws_security_group.alb.id]
  subnets         = [aws_subnet.private.id]
}

resource "aws_lb_target_group" "app" {
  name     = "shop-app"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id
  health_check {
    path = "/health"
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.web.arn
  port              = 443
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

resource "aws_launch_template" "app" {
  name_prefix            = "shop-app-"
  vpc_security_group_ids = [aws_security_group.app.id]
  user_data = base64encode(<<-EOT
    DB_HOST=${aws_db_instance.orders.address}
    REDIS_HOST=${aws_elasticache_replication_group.sessions.primary_endpoint_address}
    QUEUE_URL=${aws_sqs_queue.orders.url}
    ASSETS_BUCKET=${aws_s3_bucket.assets.id}
  EOT
  )
}

resource "aws_autoscaling_group" "app" {
  min_size            = 2
  max_size            = 6
  vpc_zone_identifier = [aws_subnet.private.id]
  target_group_arns   = [aws_lb_target_group.app.arn]
  launch_template {
    id      = aws_launch_template.app.id
    version = "$Latest"
  }
}

resource "aws_elasticache_replication_group" "sessions" {
  replication_group_id = "shop-sessions"
  description          = "Session cache"
  node_type            = "cache.t4g.small"
  num_cache_clusters   = 2
  security_group_ids   = [aws_security_group.data.id]
}

resource "aws_db_instance" "orders" {
  identifier             = "shop-orders"
  engine                 = "postgres"
  instance_class         = "db.t4g.medium"
  multi_az               = true
  vpc_security_group_ids = [aws_security_group.data.id]
}

resource "aws_sqs_queue" "orders" {
  name = "shop-orders"
}

resource "aws_lambda_function" "fulfilment" {
  function_name = "shop-fulfilment"
  runtime       = "python3.12"
  handler       = "main.handler"
  environment {
    variables = {
      DB_HOST = aws_db_instance.orders.address
    }
  }
}

resource "aws_lambda_event_source_mapping" "orders" {
  event_source_arn = aws_sqs_queue.orders.arn
  function_name    = aws_lambda_function.fulfilment.arn
}

resource "aws_s3_bucket" "assets" {
  bucket = "shop-assets-example"
}

resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "shop-alb-5xx"
  metric_name         = "HTTPCode_ELB_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 10
  evaluation_periods  = 2
  period              = 60
  statistic           = "Sum"
  dimensions = {
    LoadBalancer = aws_lb.web.arn_suffix
  }
}

resource "aws_cloudwatch_metric_alarm" "db_cpu" {
  alarm_name          = "shop-db-cpu"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  comparison_operator = "GreaterThanThreshold"
  threshold           = 80
  evaluation_periods  = 3
  period              = 60
  statistic           = "Average"
  dimensions = {
    DBInstanceIdentifier = aws_db_instance.orders.identifier
  }
}
