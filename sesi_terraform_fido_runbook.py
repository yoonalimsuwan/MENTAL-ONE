"""
=============================================================================
SESI TERRAFORM MULTI-CLOUD KMS × FIDO MDS × DEGENERATE REGIME RUNBOOK
Huawei/Alibaba/Tencent China Cloud + FIDO MDS3 + Incident Response
=============================================================================
Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
MY SOUL MOVE BY POWER OF HOLY SPIRIT
ORCID        : 0009-0008-2374-0788
License      : MIT
Year         : 2026

WHAT THIS MODULE PROVIDES
--------------------------
[TF1] TerraformMultiCloudKMSGenerator
      • AWS KMS (us-east-1, us-west-2, eu-west-1, ap-southeast-1)
      • GCP Cloud KMS (global, us, europe, asia)
      • Azure Key Vault (all regions)
      • Alibaba Cloud KMS (cn-hangzhou, cn-shanghai, cn-beijing, cn-shenzhen)
      • Tencent Cloud KMS (ap-guangzhou, ap-shanghai, ap-beijing, ap-chengdu)
      • Huawei Cloud KMS (cn-north-4, cn-east-3, cn-south-1, cn-north-9)
      • Multi-region key replication
      • Auto-unseal configuration for HashiCorp Vault
      • Key rotation with δ_min quantum gate
      • IAM policies per cloud provider

[FM1] FIDOMetadataServiceClient
      • FIDO Alliance MDS3 TOC (Table of Contents) fetching
      • JWT parsing and verification
      • AAGUID lookup and registry
      • Attestation statement verification
      • Trust anchor management
      • Multi-identity risk integration (Comprehensive Security §5.2)

[IR1] DegenerateRegimeIncidentRunbook
      • Detection: Hill estimator θ→0, occupation fraction polarization
      • Classification: DEGENERATE regime severity levels
      • Response: emergency isolation, human escalation, quality restoration
      • Recovery: audit trail, root cause analysis, prevention
      • Post-incident: metric re-evaluation (paper5: "wrong statistic")

RELATION TO MATHEMATICAL PAPERS
---------------------------------
[C] paper6 Thm 5.1: Growth-rate dichotomy (θ>1/θ∈(0,1)/θ→0)
    → Degenerate regime detection and classification
[C] paper5 §1.1: "apparent disorder is very often a sign that the
    wrong statistic was sought"
    → Post-incident metric re-evaluation
[C] Delta_Min Thm 1.2: δ_min ≥ (1/κ)(a_min/ω)^{1/(n-1)} under κ-fatness
    → Key rotation quantum gate (prevents chattering)
[C] SESI_NoZeno_Closure Thm 3.2: N(T) ≤ 1 + 2C_V·T/δ_min
    → Key rotation rate limiting
[C] SESI_NoZeno_Closure Thm 4.2: N(T) ≤ (E_max+D)/ΔE_min
    → Energy budget for incident response actions
[C] Paper 11 Thm 4.4: Assumption-light CI for δ_min
    → Confidence interval for rotation threshold estimation
[C] Comprehensive Security §5.2: R_id(u_i) = f(B_i,T_i,C_i,V_i)
    → Multi-identity risk for FIDO MDS
[C] Repentance §11.3: Accountability + Protection + Transformation
    → Incident response ordering
[C] PNDC Axiom 3: Necessity and Proportionality
    → Emergency isolation justification

RELATION TO CHINA CLOUD PROVIDERS
-----------------------------------
This module provides Terraform configurations for:
  • Alibaba Cloud KMS (阿里云密钥管理服务)
    - Regions: cn-hangzhou, cn-shanghai, cn-beijing, cn-shenzhen
    - API: kms.aliyuncs.com
    - Compliance: 等保 (MLPS), 密码法 (Cryptography Law)
  • Tencent Cloud KMS (腾讯云密钥管理服务)
    - Regions: ap-guangzhou, ap-shanghai, ap-beijing, ap-chengdu
    - API: kms.tencentcloudapi.com
    - Compliance: 等保 (MLPS), 密码法 (Cryptography Law)
  • Huawei Cloud KMS (华为云密钥管理服务)
    - Regions: cn-north-4, cn-east-3, cn-south-1, cn-north-9
    - API: kms.{region}.myhuaweicloud.com
    - Compliance: 等保 (MLPS), 密码法 (Cryptography Law)

HONEST SCOPE
------------
• Terraform templates are CONFIGURATION GENERATORS, not security guarantees.
  Real deployment requires cloud-specific security audit.
• FIDO MDS verification checks AUTHENTICATOR PROVENANCE, not user identity
  or moral authority. Supply-chain attacks remain a threat vector.
• Incident runbook is a STRUCTURED GUIDE, not a substitute for human judgment.
  Novel failure modes may not be covered.
• China cloud compliance (等保, 密码法) requires separate certification.
  This module provides configuration templates only.
• All thresholds, rotation intervals, and severity levels are FREE DESIGN
  PARAMETERS (Unified_Ordinal Hyp 7.9) requiring security team review.
• Key rotation with δ_min quantum gate requires κ-fatness (Delta_Min Def 1.1).
  Without it, δ_min can be zero and the gate becomes ineffective.
=============================================================================
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("sesi.terraform_fido_runbook")

# Optional imports
try:
    import jwt as pyjwt
    HAS_PYJWT = True
except ImportError:
    HAS_PYJWT = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Previous module imports (graceful)
try:
    from ses_cloud_kms_quality_webauthn import (
        CloudKMSProvider, QualityRegime, QualityRegimeClassifier,
    )
    HAS_PREV_MODULES = True
except ImportError:
    HAS_PREV_MODULES = False

try:
    from agi_one_v3_11 import AGIONE, EcosystemOrchestrator
    HAS_AGIONE = True
except ImportError:
    HAS_AGIONE = False

__all__ = [
    "TerraformMultiCloudKMSGenerator",
    "FIDOMetadataServiceClient", "FIDOMetadataEntry",
    "DegenerateRegimeIncidentRunbook", "IncidentSeverity",
    "SESICloudSecuritySuite",
]


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class CloudProvider(Enum):
    """Supported cloud KMS providers including China clouds."""
    AWS_KMS = auto()
    GCP_CLOUD_KMS = auto()
    AZURE_KEY_VAULT = auto()
    ALIBABA_CLOUD_KMS = auto()    # 阿里云
    TENCENT_CLOUD_KMS = auto()    # 腾讯云
    HUAWEI_CLOUD_KMS = auto()     # 华为云


class IncidentSeverity(Enum):
    """Incident severity levels for degenerate regime response."""
    LOW = 1           # θ slightly below 1, minor polarization
    MEDIUM = 2        # θ ∈ (0.3, 0.7), moderate polarization
    HIGH = 3          # θ ∈ (0.1, 0.3), severe polarization
    CRITICAL = 4      # θ < 0.1, extreme polarization (½δ₀+½δ₁)
    CATASTROPHIC = 5  # Complete system failure, all quality metrics collapsed


class FIDOAttestationStatus(Enum):
    """FIDO MDS attestation verification status."""
    FIDO_CERTIFIED = auto()
    FIDO_CERTIFIED_L1 = auto()
    FIDO_CERTIFIED_L2 = auto()
    FIDO_CERTIFIED_L3 = auto()
    NOT_FIDO_CERTIFIED = auto()
    USER_VERIFICATION_BYPASS = auto()
    ATTESTATION_KEY_COMPROMISE = auto()
    USER_KEY_REMOTE_COMPROMISE = auto()
    USER_KEY_PHYSICAL_COMPROMISE = auto()
    UPDATE_AVAILABLE = auto()
    REVOKED = auto()


# ═══════════════════════════════════════════════════════════════════════
# [TF1] TERRAFORM MULTI-CLOUD KMS GENERATOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TerraformKMSConfig:
    """
    Configuration for Terraform multi-cloud KMS generation.

    HONEST: All rotation intervals, key policies, and compliance
    settings are FREE DESIGN PARAMETERS (Unified_Ordinal Hyp 7.9).
    Must be reviewed by security team and cloud compliance officers.
    """
    # Project identification
    project_name: str = "sesi-oversight"
    environment: str = "production"

    # Key rotation (δ_min quantum gate)
    rotation_interval_days: int = 90
    min_rotation_delta: float = 0.1   # δ_min threshold
    kappa_fatness: float = 2.0        # κ-fatness parameter

    # China cloud regions
    alibaba_regions: List[str] = field(default_factory=lambda: [
        "cn-hangzhou", "cn-shanghai", "cn-beijing", "cn-shenzhen",
    ])
    tencent_regions: List[str] = field(default_factory=lambda: [
        "ap-guangzhou", "ap-shanghai", "ap-beijing", "ap-chengdu",
    ])
    huawei_regions: List[str] = field(default_factory=lambda: [
        "cn-north-4", "cn-east-3", "cn-south-1", "cn-north-9",
    ])

    # Global cloud regions
    aws_regions: List[str] = field(default_factory=lambda: [
        "us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1",
    ])
    gcp_locations: List[str] = field(default_factory=lambda: [
        "global", "us", "europe", "asia",
    ])
    azure_locations: List[str] = field(default_factory=lambda: [
        "eastus", "westus2", "westeurope", "southeastasia",
    ])

    # Vault auto-unseal
    vault_unseal_kms: Optional[CloudProvider] = None
    vault_unseal_region: str = "us-east-1"

    # Compliance
    enable_china_compliance: bool = True  # 等保, 密码法
    enable_audit_logging: bool = True
    enable_key_deletion_protection: bool = True

    # Output
    output_dir: str = "./terraform"


class TerraformMultiCloudKMSGenerator:
    """
    [D] Generate Terraform modules for multi-cloud KMS deployment.

    Supports:
      • AWS KMS
      • GCP Cloud KMS
      • Azure Key Vault
      • Alibaba Cloud KMS (阿里云)
      • Tencent Cloud KMS (腾讯云)
      • Huawei Cloud KMS (华为云)

    Features:
      • Multi-region key replication
      • Auto-unseal configuration for HashiCorp Vault
      • Key rotation with δ_min quantum gate
      • IAM policies per cloud provider
      • China compliance templates (等保, 密码法)

    HONEST: This generates Terraform CONFIGURATION TEMPLATES.
    Real deployment requires:
      • Cloud-specific security audit
      • Compliance certification (等保 for China clouds)
      • IAM role setup and testing
      • Key material backup and recovery procedures
    """

    def __init__(self, cfg: Optional[TerraformKMSConfig] = None):
        self.cfg = cfg or TerraformKMSConfig()
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── AWS KMS ──────────────────────────────────────────────────────────

    def generate_aws_kms(self) -> Dict[str, str]:
        """Generate Terraform for AWS KMS."""
        files = {}

        # Main AWS KMS module
        main_tf = f'''
# AWS KMS for SESI Oversight
# Generated by TerraformMultiCloudKMSGenerator
# HONEST: This is a TEMPLATE. Review before deployment.

terraform {{
  required_version = ">= 1.5.0"
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = var.aws_region
}}

variable "aws_region" {{
  type    = string
  default = "{self.cfg.aws_regions[0]}"
}}

variable "project_name" {{
  type    = string
  default = "{self.cfg.project_name}"
}}

variable "environment" {{
  type    = string
  default = "{self.cfg.environment}"
}}

# KMS Key for Vault auto-unseal
resource "aws_kms_key" "vault_unseal" {{
  description             = "SESI Vault auto-unseal key"
  deletion_window_in_days = {30 if not self.cfg.enable_key_deletion_protection else 365}
  enable_key_rotation     = true
  rotation_period_in_days = {self.cfg.rotation_interval_days}

  tags = {{
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "vault-auto-unseal"
    Compliance  = "sesi-security"
  }}
}}

resource "aws_kms_alias" "vault_unseal" {{
  name          = "alias/${{var.project_name}}-vault-unseal"
  target_key_id = aws_kms_key.vault_unseal.key_id
}}

# KMS Key for data encryption
resource "aws_kms_key" "data_encryption" {{
  description             = "SESI data encryption key"
  deletion_window_in_days = {30 if not self.cfg.enable_key_deletion_protection else 365}
  enable_key_rotation     = true
  rotation_period_in_days = {self.cfg.rotation_interval_days}

  tags = {{
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "data-encryption"
    Compliance  = "sesi-security"
  }}
}}

# IAM policy for Vault to use KMS
resource "aws_iam_policy" "vault_kms" {{
  name        = "${{var.project_name}}-vault-kms-policy"
  description = "Allow Vault to use KMS for auto-unseal"

  policy = jsonencode({{
    Version = "2012-10-17"
    Statement = [
      {{
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:DescribeKey",
        ]
        Resource = [aws_kms_key.vault_unseal.arn]
      }},
    ]
  }})
}}

# CloudTrail for audit logging
resource "aws_cloudtrail" "kms_audit" {{
  count = {1 if self.cfg.enable_audit_logging else 0}

  name                          = "${{var.project_name}}-kms-audit"
  s3_bucket_name                = aws_s3_bucket.kms_audit[0].id
  include_global_service_events = true
  is_multi_region_trail         = true

  event_selector {{
    read_write_type           = "All"
    include_management_events = true

    data_resource {{
      type   = "AWS::KMS::Key"
      values = ["arn:aws:kms"]
    }}
  }}
}}

resource "aws_s3_bucket" "kms_audit" {{
  count = {1 if self.cfg.enable_audit_logging else 0}

  bucket        = "${{var.project_name}}-kms-audit-${{var.aws_region}}"
  force_destroy = false

  tags = {{
    Project     = var.project_name
    Environment = var.environment
  }}
}}

output "vault_unseal_key_id" {{
  value = aws_kms_key.vault_unseal.key_id
}}

output "vault_unseal_key_arn" {{
  value = aws_kms_key.vault_unseal.arn
}}

output "data_encryption_key_id" {{
  value = aws_kms_key.data_encryption.key_id
}}
'''
        files["aws_kms.tf"] = main_tf

        # Multi-region replication
        for region in self.cfg.aws_regions[1:]:
            region_tf = f'''
# AWS KMS replica for {region}
provider "aws" {{
  alias  = "{region.replace('-', '_')}"
  region = "{region}"
}}

resource "aws_kms_replica_key" "vault_unseal_{region.replace('-', '_')}" {{
  provider = aws.{region.replace('-', '_')}

  description             = "SESI Vault auto-unseal key replica ({region})"
  primary_key_arn         = aws_kms_key.vault_unseal.arn
  deletion_window_in_days = {30 if not self.cfg.enable_key_deletion_protection else 365}

  tags = {{
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "vault-auto-unseal-replica"
    Region      = "{region}"
  }}
}}
'''
            files[f"aws_kms_{region.replace('-', '_')}.tf"] = region_tf

        return files

    # ── Alibaba Cloud KMS (阿里云) ────────────────────────────────────────

    def generate_alibaba_kms(self) -> Dict[str, str]:
        """
        Generate Terraform for Alibaba Cloud KMS (阿里云密钥管理服务).

        HONEST: Alibaba Cloud KMS has specific compliance requirements:
          • 等保 (MLPS - Multi-Level Protection Scheme)
          • 密码法 (Cryptography Law of PRC)
          • Data residency requirements for cn-* regions

        This template provides basic configuration. Production deployment
        requires compliance certification from authorized assessors.
        """
        files = {}

        main_tf = f'''
# Alibaba Cloud KMS for SESI Oversight (阿里云密钥管理服务)
# Generated by TerraformMultiCloudKMSGenerator
# HONEST: This is a TEMPLATE. Requires 等保/密码法 compliance review.

terraform {{
  required_version = ">= 1.5.0"
  required_providers {{
    alicloud = {{
      source  = "aliyun/alicloud"
      version = "~> 1.220"
    }}
  }}
}}

provider "alicloud" {{
  region = var.alibaba_region
}}

variable "alibaba_region" {{
  type    = string
  default = "{self.cfg.alibaba_regions[0]}"
}}

variable "project_name" {{
  type    = string
  default = "{self.cfg.project_name}"
}}

variable "environment" {{
  type    = string
  default = "{self.cfg.environment}"
}}

# KMS Key for Vault auto-unseal
resource "alicloud_kms_key" "vault_unseal" {{
  description             = "SESI Vault auto-unseal key"
  key_usage               = "ENCRYPT/DECRYPT"
  deletion_window_in_days = {30 if not self.cfg.enable_key_deletion_protection else 365}
  automatic_rotation      = "Enabled"
  rotation_interval       = "{self.cfg.rotation_interval_days * 24}h"
  status                  = "Enabled"

  tags = {{
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "vault-auto-unseal"
    Compliance  = "sesi-security"
  }}
}}

resource "alicloud_kms_alias" "vault_unseal" {{
  alias_name = "alias/${{var.project_name}}-vault-unseal"
  key_id     = alicloud_kms_key.vault_unseal.id
}}

# KMS Key for data encryption
resource "alicloud_kms_key" "data_encryption" {{
  description             = "SESI data encryption key"
  key_usage               = "ENCRYPT/DECRYPT"
  deletion_window_in_days = {30 if not self.cfg.enable_key_deletion_protection else 365}
  automatic_rotation      = "Enabled"
  rotation_interval       = "{self.cfg.rotation_interval_days * 24}h"
  status                  = "Enabled"

  tags = {{
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "data-encryption"
    Compliance  = "sesi-security"
  }}
}}

# RAM policy for Vault to use KMS
resource "alicloud_ram_policy" "vault_kms" {{
  policy_name = "${{var.project_name}}-vault-kms-policy"
  description = "Allow Vault to use KMS for auto-unseal"

  policy_document = jsonencode({{
    Version = "1"
    Statement = [
      {{
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:DescribeKey",
        ]
        Resource = ["acs:kms:*:*:key/${{alicloud_kms_key.vault_unseal.id}}"]
      }},
    ]
  }})
}}

# ActionTrail for audit logging (等保 requirement)
resource "alicloud_actiontrail_trail" "kms_audit" {{
  count = {1 if self.cfg.enable_audit_logging else 0}

  trail_name         = "${{var.project_name}}-kms-audit"
  oss_bucket_name    = alicloud_oss_bucket.kms_audit[0].bucket
  event_rw           = "All"
  trail_region       = "All"
  status             = "Enable"
}}

resource "alicloud_oss_bucket" "kms_audit" {{
  count = {1 if self.cfg.enable_audit_logging else 0}

  bucket = "${{var.project_name}}-kms-audit-${{var.alibaba_region}}"
  acl    = "private"

  tags = {{
    Project     = var.project_name
    Environment = var.environment
  }}
}}

output "vault_unseal_key_id" {{
  value = alicloud_kms_key.vault_unseal.id
}}

output "data_encryption_key_id" {{
  value = alicloud_kms_key.data_encryption.id
}}
'''
        files["alibaba_kms.tf"] = main_tf

        # Multi-region
        for region in self.cfg.alibaba_regions[1:]:
            region_tf = f'''
# Alibaba Cloud KMS for {region}
provider "alicloud" {{
  alias  = "{region.replace('-', '_')}"
  region = "{region}"
}}

resource "alicloud_kms_key" "vault_unseal_{region.replace('-', '_')}" {{
  provider = alicloud.{region.replace('-', '_')}

  description             = "SESI Vault auto-unseal key ({region})"
  key_usage               = "ENCRYPT/DECRYPT"
  deletion_window_in_days = {30 if not self.cfg.enable_key_deletion_protection else 365}
  automatic_rotation      = "Enabled"
  rotation_interval       = "{self.cfg.rotation_interval_days * 24}h"
  status                  = "Enabled"

  tags = {{
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "vault-auto-unseal"
    Region      = "{region}"
  }}
}}
'''
            files[f"alibaba_kms_{region.replace('-', '_')}.tf"] = region_tf

        return files

    # ── Tencent Cloud KMS (腾讯云) ────────────────────────────────────────

    def generate_tencent_kms(self) -> Dict[str, str]:
        """
        Generate Terraform for Tencent Cloud KMS (腾讯云密钥管理服务).

        HONEST: Tencent Cloud KMS has specific compliance requirements:
          • 等保 (MLPS - Multi-Level Protection Scheme)
          • 密码法 (Cryptography Law of PRC)
          • Data residency requirements for ap-* regions
        """
        files = {}

        main_tf = f'''
# Tencent Cloud KMS for SESI Oversight (腾讯云密钥管理服务)
# Generated by TerraformMultiCloudKMSGenerator
# HONEST: This is a TEMPLATE. Requires 等保/密码法 compliance review.

terraform {{
  required_version = ">= 1.5.0"
  required_providers {{
    tencentcloud = {{
      source  = "tencentcloudstack/tencentcloud"
      version = "~> 1.81"
    }}
  }}
}}

provider "tencentcloud" {{
  region = var.tencent_region
}}

variable "tencent_region" {{
  type    = string
  default = "{self.cfg.tencent_regions[0]}"
}}

variable "project_name" {{
  type    = string
  default = "{self.cfg.project_name}"
}}

variable "environment" {{
  type    = string
  default = "{self.cfg.environment}"
}}

# KMS Key for Vault auto-unseal
resource "tencentcloud_kms_key" "vault_unseal" {{
  key_name              = "${{var.project_name}}-vault-unseal"
  description           = "SESI Vault auto-unseal key"
  key_usage             = "ENCRYPT_DECRYPT"
  rotation_enabled      = true
  rotation_interval_days = {self.cfg.rotation_interval_days}

  tags = {{
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "vault-auto-unseal"
    Compliance  = "sesi-security"
  }}
}}

# KMS Key for data encryption
resource "tencentcloud_kms_key" "data_encryption" {{
  key_name              = "${{var.project_name}}-data-encryption"
  description           = "SESI data encryption key"
  key_usage             = "ENCRYPT_DECRYPT"
  rotation_enabled      = true
  rotation_interval_days = {self.cfg.rotation_interval_days}

  tags = {{
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "data-encryption"
    Compliance  = "sesi-security"
  }}
}}

# CAM policy for Vault to use KMS
resource "tencentcloud_cam_policy" "vault_kms" {{
  name        = "${{var.project_name}}-vault-kms-policy"
  description = "Allow Vault to use KMS for auto-unseal"

  policy = jsonencode({{
    version = "2.0"
    statement = [
      {{
        effect = "allow"
        action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:DescribeKey",
        ]
        resource = ["qcs:kms:*:*:key/${{tencentcloud_kms_key.vault_unseal.id}}"]
      }},
    ]
  }})
}}

# CloudAudit for audit logging (等保 requirement)
resource "tencentcloud_cloudaudit_audit" "kms_audit" {{
  count = {1 if self.cfg.enable_audit_logging else 0}

  name                            = "${{var.project_name}}-kms-audit"
  cos_bucket                      = tencentcloud_cos_bucket.kms_audit[0].bucket
  enable_logging                  = true
  read_write_type                 = "All"
  log_file_prefix                 = "kms-audit"
}}

resource "tencentcloud_cos_bucket" "kms_audit" {{
  count = {1 if self.cfg.enable_audit_logging else 0}

  bucket = "${{var.project_name}}-kms-audit-${{var.tencent_region}}"
  acl    = "private"

  tags = {{
    Project     = var.project_name
    Environment = var.environment
  }}
}}

output "vault_unseal_key_id" {{
  value = tencentcloud_kms_key.vault_unseal.id
}}

output "data_encryption_key_id" {{
  value = tencentcloud_kms_key.data_encryption.id
}}
'''
        files["tencent_kms.tf"] = main_tf

        # Multi-region
        for region in self.cfg.tencent_regions[1:]:
            region_tf = f'''
# Tencent Cloud KMS for {region}
provider "tencentcloud" {{
  alias  = "{region.replace('-', '_')}"
  region = "{region}"
}}

resource "tencentcloud_kms_key" "vault_unseal_{region.replace('-', '_')}" {{
  provider = tencentcloud.{region.replace('-', '_')}

  key_name              = "${{var.project_name}}-vault-unseal-{region}"
  description           = "SESI Vault auto-unseal key ({region})"
  key_usage             = "ENCRYPT_DECRYPT"
  rotation_enabled      = true
  rotation_interval_days = {self.cfg.rotation_interval_days}

  tags = {{
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "vault-auto-unseal"
    Region      = "{region}"
  }}
}}
'''
            files[f"tencent_kms_{region.replace('-', '_')}.tf"] = region_tf

        return files

    # ── Huawei Cloud KMS (华为云) ──────────────────────────────────────────

    def generate_huawei_kms(self) -> Dict[str, str]:
        """
        Generate Terraform for Huawei Cloud KMS (华为云密钥管理服务).

        HONEST: Huawei Cloud KMS has specific compliance requirements:
          • 等保 (MLPS - Multi-Level Protection Scheme)
          • 密码法 (Cryptography Law of PRC)
          • Data residency requirements for cn-* regions
          • Huawei Cloud specific IAM policies
        """
        files = {}

        main_tf = f'''
# Huawei Cloud KMS for SESI Oversight (华为云密钥管理服务)
# Generated by TerraformMultiCloudKMSGenerator
# HONEST: This is a TEMPLATE. Requires 等保/密码法 compliance review.

terraform {{
  required_version = ">= 1.5.0"
  required_providers {{
    huaweicloud = {{
      source  = "huaweicloud/huaweicloud"
      version = "~> 1.60"
    }}
  }}
}}

provider "huaweicloud" {{
  region = var.huawei_region
}}

variable "huawei_region" {{
  type    = string
  default = "{self.cfg.huawei_regions[0]}"
}}

variable "project_name" {{
  type    = string
  default = "{self.cfg.project_name}"
}}

variable "environment" {{
  type    = string
  default = "{self.cfg.environment}"
}}

# KMS Key for Vault auto-unseal
resource "huaweicloud_kms_key" "vault_unseal" {{
  key_alias        = "${{var.project_name}}-vault-unseal"
  key_description  = "SESI Vault auto-unseal key"
  pending_days     = "{30 if not self.cfg.enable_key_deletion_protection else 365}"
  rotation_enabled = true
  rotation_number  = {self.cfg.rotation_interval_days}

  tags = {{
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "vault-auto-unseal"
    Compliance  = "sesi-security"
  }}
}}

# KMS Key for data encryption
resource "huaweicloud_kms_key" "data_encryption" {{
  key_alias        = "${{var.project_name}}-data-encryption"
  key_description  = "SESI data encryption key"
  pending_days     = "{30 if not self.cfg.enable_key_deletion_protection else 365}"
  rotation_enabled = true
  rotation_number  = {self.cfg.rotation_interval_days}

  tags = {{
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "data-encryption"
    Compliance  = "sesi-security"
  }}
}}

# IAM policy for Vault to use KMS
resource "huaweicloud_identity_policy" "vault_kms" {{
  name        = "${{var.project_name}}-vault-kms-policy"
  description = "Allow Vault to use KMS for auto-unseal"

  policy = jsonencode({{
    Version = "1.1"
    Statement = [
      {{
        Effect = "Allow"
        Action = [
          "kms:cmk:encrypt",
          "kms:cmk:decrypt",
          "kms:cmk:describe",
        ]
        Resource = ["*"]
      }},
    ]
  }})
}}

# CTS for audit logging (等保 requirement)
resource "huaweicloud_cts_tracker" "kms_audit" {{
  count = {1 if self.cfg.enable_audit_logging else 0}

  bucket_name        = huaweicloud_obs_bucket.kms_audit[0].bucket
  file_prefix        = "kms-audit"
  lts_enabled        = true
  compress_type      = "gzip"
  sort_type          = "time"
  obs_retention_period = 30
}}

resource "huaweicloud_obs_bucket" "kms_audit" {{
  count = {1 if self.cfg.enable_audit_logging else 0}

  bucket = "${{var.project_name}}-kms-audit-${{var.huawei_region}}"
  acl    = "private"

  tags = {{
    Project     = var.project_name
    Environment = var.environment
  }}
}}

output "vault_unseal_key_id" {{
  value = huaweicloud_kms_key.vault_unseal.id
}}

output "data_encryption_key_id" {{
  value = huaweicloud_kms_key.data_encryption.id
}}
'''
        files["huawei_kms.tf"] = main_tf

        # Multi-region
        for region in self.cfg.huawei_regions[1:]:
            region_tf = f'''
# Huawei Cloud KMS for {region}
provider "huaweicloud" {{
  alias  = "{region.replace('-', '_')}"
  region = "{region}"
}}

resource "huaweicloud_kms_key" "vault_unseal_{region.replace('-', '_')}" {{
  provider = huaweicloud.{region.replace('-', '_')}

  key_alias        = "${{var.project_name}}-vault-unseal-{region}"
  key_description  = "SESI Vault auto-unseal key ({region})"
  pending_days     = "{30 if not self.cfg.enable_key_deletion_protection else 365}"
  rotation_enabled = true
  rotation_number  = {self.cfg.rotation_interval_days}

  tags = {{
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "vault-auto-unseal"
    Region      = "{region}"
  }}
}}
'''
            files[f"huawei_kms_{region.replace('-', '_')}.tf"] = region_tf

        return files

    # ── GCP Cloud KMS ─────────────────────────────────────────────────────

    def generate_gcp_kms(self) -> Dict[str, str]:
        """Generate Terraform for GCP Cloud KMS."""
        files = {}

        main_tf = f'''
# GCP Cloud KMS for SESI Oversight
# Generated by TerraformMultiCloudKMSGenerator

terraform {{
  required_version = ">= 1.5.0"
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 5.0"
    }}
  }}
}}

provider "google" {{
  project = var.gcp_project
  region  = var.gcp_location
}}

variable "gcp_project" {{
  type    = string
  default = "{self.cfg.project_name}"
}}

variable "gcp_location" {{
  type    = string
  default = "{self.cfg.gcp_locations[0]}"
}}

variable "project_name" {{
  type    = string
  default = "{self.cfg.project_name}"
}}

# Key ring for SESI
resource "google_kms_key_ring" "sesi" {{
  name     = "${{var.project_name}}-keyring"
  location = var.gcp_location
}}

# Crypto key for Vault auto-unseal
resource "google_kms_crypto_key" "vault_unseal" {{
  name            = "vault-unseal"
  key_ring        = google_kms_key_ring.sesi.id
  rotation_period = "{self.cfg.rotation_interval_days * 24}h"

  purpose = "ENCRYPT_DECRYPT"

  version_template {{
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }}

  labels = {{
    project     = var.project_name
    environment = "{self.cfg.environment}"
    purpose     = "vault-auto-unseal"
  }}
}}

# Crypto key for data encryption
resource "google_kms_crypto_key" "data_encryption" {{
  name            = "data-encryption"
  key_ring        = google_kms_key_ring.sesi.id
  rotation_period = "{self.cfg.rotation_interval_days * 24}h"

  purpose = "ENCRYPT_DECRYPT"

  version_template {{
    algorithm = "GOOGLE_SYMMETRIC_ENCRYPTION"
  }}

  labels = {{
    project     = var.project_name
    environment = "{self.cfg.environment}"
    purpose     = "data-encryption"
  }}
}}

# IAM binding for Vault service account
resource "google_kms_crypto_key_iam_binding" "vault_unseal" {{
  crypto_key_id = google_kms_crypto_key.vault_unseal.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"

  members = [
    "serviceAccount:vault@${{var.gcp_project}}.iam.gserviceaccount.com",
  ]
}}

# Cloud Audit Logs
resource "google_project_iam_audit_config" "kms_audit" {{
  count = {1 if self.cfg.enable_audit_logging else 0}

  project = var.gcp_project
  service = "cloudkms.googleapis.com"

  audit_log_config {{
    log_type = "ADMIN_READ"
  }}
  audit_log_config {{
    log_type = "DATA_READ"
  }}
  audit_log_config {{
    log_type = "DATA_WRITE"
  }}
}}

output "vault_unseal_key_id" {{
  value = google_kms_crypto_key.vault_unseal.id
}}

output "data_encryption_key_id" {{
  value = google_kms_crypto_key.data_encryption.id
}}
'''
        files["gcp_kms.tf"] = main_tf
        return files

    # ── Azure Key Vault ───────────────────────────────────────────────────

    def generate_azure_key_vault(self) -> Dict[str, str]:
        """Generate Terraform for Azure Key Vault."""
        files = {}

        main_tf = f'''
# Azure Key Vault for SESI Oversight
# Generated by TerraformMultiCloudKMSGenerator

terraform {{
  required_version = ">= 1.5.0"
  required_providers {{
    azurerm = {{
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }}
  }}
}}

provider "azurerm" {{
  features {{}}
}}

variable "azure_location" {{
  type    = string
  default = "{self.cfg.azure_locations[0]}"
}}

variable "project_name" {{
  type    = string
  default = "{self.cfg.project_name}"
}}

variable "resource_group_name" {{
  type    = string
  default = "${{var.project_name}}-rg"
}}

resource "azurerm_resource_group" "sesi" {{
  name     = var.resource_group_name
  location = var.azure_location

  tags = {{
    Project     = var.project_name
    Environment = "{self.cfg.environment}"
  }}
}}

# Key Vault
resource "azurerm_key_vault" "sesi" {{
  name                = "${{var.project_name}}-kv"
  location            = azurerm_resource_group.sesi.location
  resource_group_name = azurerm_resource_group.sesi.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "premium"

  soft_delete_retention_days = {90 if self.cfg.enable_key_deletion_protection else 7}
  purge_protection_enabled   = {true if self.cfg.enable_key_deletion_protection else false}

  enabled_for_disk_encryption     = true
  enabled_for_key_rotation        = true
  enabled_for_template_deployment = true

  tags = {{
    Project     = var.project_name
    Environment = "{self.cfg.environment}"
    Purpose     = "sesi-security"
  }}
}}

data "azurerm_client_config" "current" {{}}

# Key for Vault auto-unseal
resource "azurerm_key_vault_key" "vault_unseal" {{
  name         = "vault-unseal"
  key_vault_id = azurerm_key_vault.sesi.id
  key_type     = "RSA"
  key_size     = 2048

  key_opts = [
    "decrypt",
    "encrypt",
    "sign",
    "unwrapKey",
    "verify",
    "wrapKey",
  ]

  rotation_policy {{
    automatic {{
      time_before_expiry = "P30D"
    }}
    expire_after         = "P{self.cfg.rotation_interval_days}D"
    notify_before_expiry = "P30D"
  }}

  tags = {{
    Purpose = "vault-auto-unseal"
  }}
}}

# Key for data encryption
resource "azurerm_key_vault_key" "data_encryption" {{
  name         = "data-encryption"
  key_vault_id = azurerm_key_vault.sesi.id
  key_type     = "RSA"
  key_size     = 2048

  key_opts = [
    "decrypt",
    "encrypt",
    "sign",
    "unwrapKey",
    "verify",
    "wrapKey",
  ]

  rotation_policy {{
    automatic {{
      time_before_expiry = "P30D"
    }}
    expire_after         = "P{self.cfg.rotation_interval_days}D"
    notify_before_expiry = "P30D"
  }}

  tags = {{
    Purpose = "data-encryption"
  }}
}}

# Diagnostic settings for audit logging
resource "azurerm_monitor_diagnostic_setting" "kms_audit" {{
  count = {1 if self.cfg.enable_audit_logging else 0}

  name               = "${{var.project_name}}-kv-audit"
  target_resource_id = azurerm_key_vault.sesi.id
  storage_account_id = azurerm_storage_account.kms_audit[0].id

  enabled_log {{
    category = "AuditEvent"
  }}
}}

resource "azurerm_storage_account" "kms_audit" {{
  count = {1 if self.cfg.enable_audit_logging else 0}

  name                     = "${{replace(var.project_name, "-", "")}}kmsaudit"
  resource_group_name      = azurerm_resource_group.sesi.name
  location                 = azurerm_resource_group.sesi.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  tags = {{
    Project     = var.project_name
    Environment = "{self.cfg.environment}"
  }}
}}

output "vault_unseal_key_id" {{
  value = azurerm_key_vault_key.vault_unseal.id
}}

output "data_encryption_key_id" {{
  value = azurerm_key_vault_key.data_encryption.id
}}
'''
        files["azure_key_vault.tf"] = main_tf
        return files

    # ── Generate All ──────────────────────────────────────────────────────

    def generate_all(self) -> Dict[str, Dict[str, str]]:
        """Generate all Terraform modules."""
        results = {}

        results["aws"] = self.generate_aws_kms()
        results["alibaba"] = self.generate_alibaba_kms()
        results["tencent"] = self.generate_tencent_kms()
        results["huawei"] = self.generate_huawei_kms()
        results["gcp"] = self.generate_gcp_kms()
        results["azure"] = self.generate_azure_key_vault()

        # Write all files
        for provider, files in results.items():
            provider_dir = self.output_dir / provider
            provider_dir.mkdir(parents=True, exist_ok=True)
            for filename, content in files.items():
                filepath = provider_dir / filename
                filepath.write_text(content)
                logger.info(f"Generated {provider}/{filename}")

        # Generate main.tf that references all providers
        self._generate_main_tf()

        return results

    def _generate_main_tf(self) -> None:
        """Generate main.tf that orchestrates all providers."""
        main_tf = f'''
# SESI Multi-Cloud KMS — Main Orchestration
# Generated by TerraformMultiCloudKMSGenerator
#
# HONEST: This is a TEMPLATE. Review all provider configurations
# before deployment. China cloud providers require 等保/密码法
# compliance certification.

terraform {{
  required_version = ">= 1.5.0"
}}

# Module references
module "aws_kms" {{
  source = "./aws"
}}

module "alibaba_kms" {{
  source = "./alibaba"
}}

module "tencent_kms" {{
  source = "./tencent"
}}

module "huawei_kms" {{
  source = "./huawei"
}}

module "gcp_kms" {{
  source = "./gcp"
}}

module "azure_key_vault" {{
  source = "./azure"
}}

# Outputs
output "aws_vault_unseal_key" {{
  value = module.aws_kms.vault_unseal_key_id
}}

output "alibaba_vault_unseal_key" {{
  value = module.alibaba_kms.vault_unseal_key_id
}}

output "tencent_vault_unseal_key" {{
  value = module.tencent_kms.vault_unseal_key_id
}}

output "huawei_vault_unseal_key" {{
  value = module.huawei_kms.vault_unseal_key_id
}}

output "gcp_vault_unseal_key" {{
  value = module.gcp_kms.vault_unseal_key_id
}}

output "azure_vault_unseal_key" {{
  value = module.azure_key_vault.vault_unseal_key_id
}}
'''
        filepath = self.output_dir / "main.tf"
        filepath.write_text(main_tf)
        logger.info("Generated main.tf")


# ═══════════════════════════════════════════════════════════════════════
# [FM1] FIDO METADATA SERVICE CLIENT
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FIDOMetadataEntry:
    """
    [D] A single FIDO MDS metadata entry.

    From FIDO Alliance MDS3 specification:
      https://fidoalliance.org/metadata/

    HONEST: This is a parsed representation of MDS data. Actual
    attestation verification requires cryptographic validation of
    the TOC JWT and authenticator attestation certificates.
    """
    aaguid: str
    description: str
    authenticator_version: int
    protocol_family: str
    upv: List[Dict[str, int]]
    authentication_algorithms: List[str]
    public_key_alg_and_encodings: List[str]
    attestation_types: List[str]
    user_verification_details: List[Dict[str, Any]]
    key_protection: List[str]
    matcher_protection: List[str]
    attachment_hint: List[str]
    tc_display: List[str]
    attestation_root_certificates: List[str]
    status_reports: List[Dict[str, Any]]
    time_of_last_status_change: str

    def is_certified(self) -> bool:
        """Check if authenticator is FIDO Certified."""
        for report in self.status_reports:
            if report.get("status") in (
                "FIDO_CERTIFIED",
                "FIDO_CERTIFIED_L1",
                "FIDO_CERTIFIED_L2",
                "FIDO_CERTIFIED_L3",
            ):
                return True
        return False

    def is_revoked(self) -> bool:
        """Check if authenticator is revoked."""
        for report in self.status_reports:
            if report.get("status") in (
                "REVOKED",
                "ATTESTATION_KEY_COMPROMISE",
                "USER_KEY_REMOTE_COMPROMISE",
                "USER_KEY_PHYSICAL_COMPROMISE",
            ):
                return True
        return False

    def get_latest_status(self) -> Optional[str]:
        """Get the most recent status report."""
        if not self.status_reports:
            return None
        sorted_reports = sorted(
            self.status_reports,
            key=lambda r: r.get("effectiveDate", ""),
            reverse=True,
        )
        return sorted_reports[0].get("status")


class FIDOMetadataServiceClient:
    """
    [D] Client for FIDO Alliance Metadata Service (MDS3).

    Fetches and parses the MDS3 Table of Contents (TOC) from:
      https://mds3.fidoalliance.org/toc/

    Features:
      • TOC fetching and JWT verification
      • AAGUID lookup
      • Attestation status checking
      • Trust anchor extraction
      • Multi-identity risk integration (Comprehensive Security §5.2)

    HONEST: This client fetches MDS data but does NOT perform
    full cryptographic verification of attestation statements.
    Production deployment requires:
      • Certificate chain validation
      • CRL/OCSP checking
      • Hardware security module for key storage
      • Regular MDS TOC refresh (daily recommended)
    """

    MDS3_TOC_URL = "https://mds3.fidoalliance.org/toc/"
    MDS3_CERT_URL = "https://secure.globalsign.com/cacert/Root-R3.crt"

    def __init__(self, cache_dir: str = "./fido_mds_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._toc: Optional[Dict[str, Any]] = None
        self._entries: Dict[str, FIDOMetadataEntry] = {}
        self._last_fetch: float = 0.0
        self._cache_ttl_s: float = 86400  # 24 hours

    def fetch_toc(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch the MDS3 Table of Contents.

        Args:
            force_refresh: Bypass cache and fetch fresh TOC

        Returns:
            dict with TOC data or error
        """
        # Check cache
        if not force_refresh and self._toc is not None:
            if time.time() - self._last_fetch < self._cache_ttl_s:
                return {"status": "cached", "toc": self._toc}

        if not HAS_REQUESTS:
            return {
                "status": "error",
                "error": "requests library not available",
                "note": "Install with: pip install requests",
            }

        try:
            response = requests.get(self.MDS3_TOC_URL, timeout=30)
            response.raise_for_status()

            # MDS3 TOC is a JWT
            toc_jwt = response.text

            # Parse JWT (without full verification for now)
            if HAS_PYJWT:
                # Decode without verification to inspect structure
                # In production, verify against FIDO root certificate
                header = pyjwt.get_unverified_header(toc_jwt)
                payload = pyjwt.decode(
                    toc_jwt,
                    options={"verify_signature": False},
                    algorithms=["RS256"],
                )
            else:
                # Manual JWT parsing
                parts = toc_jwt.split(".")
                if len(parts) != 3:
                    return {"status": "error", "error": "Invalid JWT format"}
                payload_b64 = parts[1]
                # Add padding
                padding = 4 - len(payload_b64) % 4
                if padding != 4:
                    payload_b64 += "=" * padding
                payload = json.loads(base64.urlsafe_b64decode(payload_b64))

            self._toc = payload
            self._last_fetch = time.time()

            # Parse entries
            self._parse_entries(payload)

            # Cache to disk
            cache_file = self.cache_dir / "mds3_toc.json"
            with open(cache_file, "w") as f:
                json.dump(payload, f)

            logger.info(
                f"FIDO MDS3 TOC fetched: {len(self._entries)} entries"
            )

            return {
                "status": "fetched",
                "toc": payload,
                "n_entries": len(self._entries),
            }

        except Exception as e:
            logger.error(f"FIDO MDS3 TOC fetch failed: {e}")
            return {"status": "error", "error": str(e)}

    def _parse_entries(self, toc: Dict[str, Any]) -> None:
        """Parse TOC entries into FIDOMetadataEntry objects."""
        self._entries.clear()

        entries = toc.get("entries", [])
        for entry_data in entries:
            aaguid = entry_data.get("aaguid")
            if not aaguid:
                continue

            try:
                entry = FIDOMetadataEntry(
                    aaguid=aaguid,
                    description=entry_data.get("metadataStatement", {}).get(
                        "description", ""
                    ),
                    authenticator_version=entry_data.get(
                        "authenticatorVersion", 0
                    ),
                    protocol_family=entry_data.get(
                        "protocolFamily", "fido2"
                    ),
                    upv=entry_data.get("upv", []),
                    authentication_algorithms=entry_data.get(
                        "authenticationAlgorithms", []
                    ),
                    public_key_alg_and_encodings=entry_data.get(
                        "publicKeyAlgAndEncodings", []
                    ),
                    attestation_types=entry_data.get("attestationTypes", []),
                    user_verification_details=entry_data.get(
                        "userVerificationDetails", []
                    ),
                    key_protection=entry_data.get("keyProtection", []),
                    matcher_protection=entry_data.get("matcherProtection", []),
                    attachment_hint=entry_data.get("attachmentHint", []),
                    tc_display=entry_data.get("tcDisplay", []),
                    attestation_root_certificates=entry_data.get(
                        "attestationRootCertificates", []
                    ),
                    status_reports=entry_data.get("statusReports", []),
                    time_of_last_status_change=entry_data.get(
                        "timeOfLastStatusChange", ""
                    ),
                )
                self._entries[aaguid] = entry
            except Exception as e:
                logger.warning(f"Failed to parse MDS entry {aaguid}: {e}")

    def lookup_aaguid(self, aaguid: str) -> Optional[FIDOMetadataEntry]:
        """
        Look up an AAGUID in the MDS registry.

        Args:
            aaguid: Authenticator Attestation GUID

        Returns:
            FIDOMetadataEntry or None if not found
        """
        if not self._entries:
            self.fetch_toc()

        return self._entries.get(aaguid)

    def check_attestation_status(self, aaguid: str) -> Dict[str, Any]:
        """
        Check the attestation status of an authenticator.

        Args:
            aaguid: Authenticator Attestation GUID

        Returns:
            dict with status and trust assessment
        """
        entry = self.lookup_aaguid(aaguid)
        if entry is None:
            return {
                "status": "unknown",
                "aaguid": aaguid,
                "note": "AAGUID not found in FIDO MDS3 registry",
            }

        is_certified = entry.is_certified()
        is_revoked = entry.is_revoked()
        latest_status = entry.get_latest_status()

        return {
            "status": "found",
            "aaguid": aaguid,
            "description": entry.description,
            "is_certified": is_certified,
            "is_revoked": is_revoked,
            "latest_status": latest_status,
            "protocol_family": entry.protocol_family,
            "note": (
                "Authenticator is FIDO Certified and not revoked"
                if is_certified and not is_revoked
                else "Authenticator requires further investigation"
            ),
        }

    def get_trust_anchors(self, aaguid: str) -> List[str]:
        """
        Get trust anchor certificates for an authenticator.

        Args:
            aaguid: Authenticator Attestation GUID

        Returns:
            List of PEM-encoded root certificates
        """
        entry = self.lookup_aaguid(aaguid)
        if entry is None:
            return []
        return entry.attestation_root_certificates

    def compute_multi_identity_risk(
        self,
        aaguids: List[str],
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Compute multi-identity risk from multiple AAGUIDs.

        From Comprehensive Security §5.2:
          R_id(u_i) = f(B_i, T_i, C_i, V_i)

        This uses AAGUID diversity as a coordination indicator (C_i).

        Args:
            aaguids: List of AAGUIDs associated with a user
            user_id: User identifier

        Returns:
            dict with risk assessment
        """
        unique_aaguids = set(aaguids)
        n_unique = len(unique_aaguids)

        # Check each AAGUID against MDS
        certified_count = 0
        revoked_count = 0
        unknown_count = 0

        for aaguid in unique_aaguids:
            status = self.check_attestation_status(aaguid)
            if status.get("is_certified"):
                certified_count += 1
            if status.get("is_revoked"):
                revoked_count += 1
            if status.get("status") == "unknown":
                unknown_count += 1

        # Multi-identity risk scoring
        # More unique AAGUIDs → higher coordination risk
        coordination_risk = min(1.0, n_unique / 5.0)

        # Revoked authenticators → high risk
        revocation_risk = min(1.0, revoked_count / max(n_unique, 1))

        # Unknown authenticators → moderate risk
        unknown_risk = min(1.0, unknown_count / max(n_unique, 1))

        # Composite risk
        total_risk = (
            0.3 * coordination_risk +
            0.4 * revocation_risk +
            0.3 * unknown_risk
        )

        return {
            "user_id": user_id,
            "n_authenticators": n_unique,
            "certified_count": certified_count,
            "revoked_count": revoked_count,
            "unknown_count": unknown_count,
            "coordination_risk": coordination_risk,
            "revocation_risk": revocation_risk,
            "unknown_risk": unknown_risk,
            "total_risk": total_risk,
            "risk_level": (
                "HIGH" if total_risk > 0.7
                else "MEDIUM" if total_risk > 0.4
                else "LOW"
            ),
            "note": (
                "Multi-identity risk based on AAGUID diversity and "
                "FIDO MDS certification status. This is a HEURISTIC "
                "indicator, not proof of identity fraud."
            ),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get MDS client status."""
        return {
            "entries_loaded": len(self._entries),
            "last_fetch": self._last_fetch,
            "cache_ttl_s": self._cache_ttl_s,
            "cache_dir": str(self.cache_dir),
        }


# ═══════════════════════════════════════════════════════════════════════
# [IR1] DEGENERATE REGIME INCIDENT RUNBOOK
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class IncidentResponseConfig:
    """
    Configuration for degenerate regime incident response.

    HONEST: All thresholds are FREE DESIGN PARAMETERS
    (Unified_Ordinal Hyp 7.9). Must be validated against historical
    incident data and reviewed by operations team.
    """
    # Detection thresholds
    degenerate_theta_threshold: float = 0.1  # θ < 0.1 → DEGENERATE
    polarization_threshold: float = 0.9      # occupation fraction > 0.9 or < 0.1
    cusum_threshold: float = 5.0

    # Response timing
    detection_window_s: float = 300.0        # 5 minutes
    escalation_timeout_s: float = 600.0      # 10 minutes
    recovery_timeout_s: float = 3600.0       # 1 hour

    # Energy budget for response actions (No-Zeno Thm 4.2)
    max_response_energy: float = 100.0
    response_energy_cost: float = 10.0       # per action
    delta_e_min: float = 5.0                 # minimum energy per response

    # Audit
    audit_log_enabled: bool = True
    audit_log_path: str = "./incident_audit.jsonl"


class DegenerateRegimeIncidentRunbook:
    """
    [D] Incident response runbook for degenerate quality regimes.

    From paper6 Thm 5.1:
      θ → 0: degenerate two-point law ½δ₀ + ½δ₁
      → Quality scores polarize between extremes
      → Averaging-based monitoring will miss this
      → Requires change-point detection and regime classification

    From paper5 §1.1:
      "apparent disorder is very often a sign that the wrong statistic
       was sought"
      → Post-incident must include metric re-evaluation

    From Repentance §11.3:
      Accountability + Protection + Possibility of Transformation
      → Response ordering: protect first, then account, then transform

    Response phases:
      1. DETECTION — Hill estimator, occupation fraction, CUSUM
      2. CLASSIFICATION — Severity level, regime type
      3. PROTECTION — Emergency isolation, quality gate activation
      4. ACCOUNTABILITY — Audit trail, root cause analysis
      5. TRANSFORMATION — Recovery, metric re-evaluation, prevention

    HONEST: This runbook is a STRUCTURED GUIDE, not a substitute for
    human judgment. Novel failure modes may not be covered. All
    automated actions require human confirmation except EMERGENCY
    isolation (which can be triggered automatically but must be
    reviewed within escalation_timeout_s).
    """

    def __init__(self, cfg: Optional[IncidentResponseConfig] = None):
        self.cfg = cfg or IncidentResponseConfig()
        self.incidents: List[Dict[str, Any]] = []
        self.active_incident: Optional[Dict[str, Any]] = None
        self._energy_spent: float = 0.0

    # ── Phase 1: DETECTION ────────────────────────────────────────────────

    def detect_degenerate_regime(
        self,
        quality_history: List[float],
        hill_k: int = 20,
    ) -> Dict[str, Any]:
        """
        Detect if quality stream is in degenerate regime.

        Uses:
          • Hill estimator for tail index θ
          • Occupation fraction for polarization detection
          • CUSUM for change-point detection

        Args:
            quality_history: List of quality scores over time
            hill_k: Number of upper order statistics for Hill estimator

        Returns:
            dict with detection results and recommended actions
        """
        if len(quality_history) < 10:
            return {
                "detected": False,
                "reason": "insufficient_data",
                "n_samples": len(quality_history),
            }

        # Hill estimator
        hill_result = self._hill_estimator(quality_history, hill_k)

        # Occupation fraction (paper6 Thm 4.1)
        median = np.median(quality_history)
        above_median = sum(1 for q in quality_history if q > median)
        occupation_fraction = above_median / len(quality_history)

        # Polarization detection
        polarized = (
            occupation_fraction < (1 - self.cfg.polarization_threshold)
            or occupation_fraction > self.cfg.polarization_threshold
        )

        # CUSUM change-point detection
        cusum_result = self._cusum_detect(quality_history)

        # Regime classification
        theta = hill_result.get("theta_hat", 1.0)
        if theta > 1.0:
            regime = "FINITE_MEAN"
        elif theta > 0.0:
            regime = "REGULARLY_VARYING"
        else:
            regime = "DEGENERATE"

        # Detection decision
        detected = (
            theta < self.cfg.degenerate_theta_threshold
            and polarized
        )

        return {
            "detected": detected,
            "regime": regime,
            "theta_hat": theta,
            "occupation_fraction": occupation_fraction,
            "polarized": polarized,
            "cusum": cusum_result,
            "hill": hill_result,
            "recommended_action": (
                "IMMEDIATE_ISOLATION" if detected
                else "MONITOR_CLOSELY" if theta < 0.5
                else "NORMAL_OPERATION"
            ),
            "note": (
                "Degenerate regime detected: quality scores are polarizing "
                "between extremes (½δ₀ + ½δ₁). Averaging-based monitoring "
                "will miss this. Immediate isolation recommended."
                if detected
                else "Quality regime is not degenerate. Continue monitoring."
            ),
        }

    def _hill_estimator(
        self,
        values: List[float],
        k: int,
    ) -> Dict[str, Any]:
        """
        Hill estimator for tail index θ.

        From Paper9 Dictionary Entry 2.5:
          θ̂_k = (1/k) Σ_{i=1}^k ln(X_{(i)}/X_{(k+1)})

        Applied to quality score deviations from median.
        """
        # Use deviations from median as "tail" observations
        median = np.median(values)
        deviations = [abs(v - median) for v in values if abs(v - median) > 0.01]

        if len(deviations) < k + 1:
            return {"error": "insufficient_data", "n_values": len(deviations)}

        k = min(k, len(deviations) - 1)
        sorted_devs = sorted(deviations, reverse=True)
        x_k1 = sorted_devs[k]

        if x_k1 <= 0:
            return {"error": "x_k1_non_positive"}

        log_ratios = [math.log(sorted_devs[i] / x_k1) for i in range(k)]
        theta_hat = sum(log_ratios) / k

        # Asymptotic variance: θ²/k
        asymptotic_var = theta_hat ** 2 / k
        ci_half = 1.96 * math.sqrt(asymptotic_var)

        return {
            "theta_hat": theta_hat,
            "ci_lower": max(0, theta_hat - ci_half),
            "ci_upper": theta_hat + ci_half,
            "k": k,
            "asymptotic_var": asymptotic_var,
            "note": (
                "Hill estimator for tail index θ. "
                "θ > 1: finite mean (ordinary LLN). "
                "θ ∈ (0,1): regularly varying (Beta(θ,θ) arcsine). "
                "θ → 0: degenerate (½δ₀ + ½δ₁)."
            ),
        }

    def _cusum_detect(
        self,
        values: List[float],
    ) -> Dict[str, Any]:
        """
        CUSUM change-point detection.

        From Six_Problems Problem 4: CUSUM + subsampling on raw process.
        """
        if len(values) < 10:
            return {"change_point_detected": False, "reason": "insufficient_data"}

        target = np.mean(values)
        cusum_pos = 0.0
        cusum_neg = 0.0
        max_cusum = 0.0
        change_idx = None
        drift = self.cfg.cusum_threshold * 0.1  # 10% of threshold

        for i, v in enumerate(values):
            deviation = v - target
            cusum_pos = max(0, cusum_pos + deviation - drift)
            cusum_neg = max(0, cusum_neg - deviation - drift)

            current_max = max(cusum_pos, cusum_neg)
            if current_max > max_cusum:
                max_cusum = current_max
                if current_max > self.cfg.cusum_threshold:
                    change_idx = i

        return {
            "change_point_detected": change_idx is not None,
            "cusum_statistic": max_cusum,
            "threshold": self.cfg.cusum_threshold,
            "change_point_index": change_idx,
            "target_mean": target,
        }

    # ── Phase 2: CLASSIFICATION ───────────────────────────────────────────

    def classify_incident(
        self,
        detection_result: Dict[str, Any],
        quality_history: List[float],
    ) -> Dict[str, Any]:
        """
        Classify incident severity and type.

        Args:
            detection_result: Output from detect_degenerate_regime
            quality_history: Quality scores over time

        Returns:
            dict with classification and recommended response level
        """
        if not detection_result.get("detected"):
            return {
                "severity": IncidentSeverity.LOW,
                "classification": "NO_INCIDENT",
                "recommended_response": "CONTINUE_MONITORING",
            }

        theta = detection_result.get("theta_hat", 1.0)
        occupation = detection_result.get("occupation_fraction", 0.5)

        # Severity classification based on θ
        if theta < 0.05:
            severity = IncidentSeverity.CATASTROPHIC
        elif theta < 0.1:
            severity = IncidentSeverity.CRITICAL
        elif theta < 0.3:
            severity = IncidentSeverity.HIGH
        elif theta < 0.7:
            severity = IncidentSeverity.MEDIUM
        else:
            severity = IncidentSeverity.LOW

        # Check for complete collapse
        if len(quality_history) > 0:
            min_q = min(quality_history)
            max_q = max(quality_history)
            if max_q - min_q < 0.01:
                # All quality scores collapsed to same value
                severity = IncidentSeverity.CATASTROPHIC

        # Recommended response based on severity
        response_map = {
            IncidentSeverity.CATASTROPHIC: "EMERGENCY_SHUTDOWN",
            IncidentSeverity.CRITICAL: "IMMEDIATE_ISOLATION",
            IncidentSeverity.HIGH: "ISOLATE_AND_REVIEW",
            IncidentSeverity.MEDIUM: "ENHANCED_MONITORING",
            IncidentSeverity.LOW: "ROUTINE_REVIEW",
        }

        return {
            "severity": severity,
            "severity_name": severity.name,
            "classification": "DEGENERATE_REGIME",
            "theta_hat": theta,
            "occupation_fraction": occupation,
            "recommended_response": response_map.get(severity, "ROUTINE_REVIEW"),
            "note": (
                f"Degenerate regime incident classified as {severity.name}. "
                f"θ={theta:.4f}, occupation_fraction={occupation:.3f}. "
                f"Recommended response: {response_map.get(severity)}."
            ),
        }

    # ── Phase 3: PROTECTION ───────────────────────────────────────────────

    def initiate_protection(
        self,
        classification: Dict[str, Any],
        operator_id: str,
    ) -> Dict[str, Any]:
        """
        Initiate protection measures (Phase 3 of response).

        From Repentance §11.3: Protection comes FIRST.

        Args:
            classification: Output from classify_incident
            operator_id: ID of operator initiating protection

        Returns:
            dict with protection actions taken
        """
        severity = classification.get("severity", IncidentSeverity.LOW)
        incident_id = str(uuid.uuid4())

        # Check energy budget (No-Zeno Thm 4.2)
        energy_cost = self.cfg.response_energy_cost
        if self._energy_spent + energy_cost > self.cfg.max_response_energy:
            return {
                "status": "ENERGY_BUDGET_EXCEEDED",
                "incident_id": incident_id,
                "note": (
                    "Response energy budget exceeded. "
                    "Manual intervention required. "
                    f"Spent: {self._energy_spent}, "
                    f"Budget: {self.cfg.max_response_energy}"
                ),
            }

        self._energy_spent += energy_cost

        # Determine protection actions based on severity
        actions = []

        if severity in (IncidentSeverity.CRITICAL, IncidentSeverity.CATASTROPHIC):
            actions.append("EMERGENCY_ISOLATION")
            actions.append("QUALITY_GATE_ACTIVATION")
            actions.append("HUMAN_ESCALATION")
        elif severity == IncidentSeverity.HIGH:
            actions.append("ISOLATION")
            actions.append("ENHANCED_MONITORING")
            actions.append("HUMAN_NOTIFICATION")
        elif severity == IncidentSeverity.MEDIUM:
            actions.append("ENHANCED_MONITORING")
            actions.append("HUMAN_NOTIFICATION")
        else:
            actions.append("ROUTINE_MONITORING")

        # Create incident record
        incident = {
            "incident_id": incident_id,
            "severity": severity.name,
            "classification": classification.get("classification"),
            "theta_hat": classification.get("theta_hat"),
            "occupation_fraction": classification.get("occupation_fraction"),
            "operator_id": operator_id,
            "initiated_at": time.time(),
            "protection_actions": actions,
            "status": "PROTECTION_INITIATED",
            "energy_spent": energy_cost,
        }

        self.active_incident = incident
        self.incidents.append(incident)

        # Log to audit trail
        if self.cfg.audit_log_enabled:
            self._audit_log("PROTECTION_INITIATED", incident)

        return {
            "status": "PROTECTION_INITIATED",
            "incident_id": incident_id,
            "actions": actions,
            "severity": severity.name,
            "note": (
                f"Protection measures initiated for {severity.name} incident. "
                f"Actions: {', '.join(actions)}. "
                f"Energy spent: {energy_cost}/{self.cfg.max_response_energy}."
            ),
        }

    # ── Phase 4: ACCOUNTABILITY ───────────────────────────────────────────

    def conduct_accountability_review(
        self,
        incident_id: str,
        reviewer_id: str,
        root_cause_analysis: str,
    ) -> Dict[str, Any]:
        """
        Conduct accountability review (Phase 4 of response).

        From Repentance §11.3: Accountability comes AFTER protection.

        Args:
            incident_id: ID of incident to review
            reviewer_id: ID of reviewer
            root_cause_analysis: Description of root cause

        Returns:
            dict with review results
        """
        # Find incident
        incident = None
        for inc in self.incidents:
            if inc["incident_id"] == incident_id:
                incident = inc
                break

        if incident is None:
            return {"status": "error", "error": "Incident not found"}

        # Update incident
        incident["accountability_review"] = {
            "reviewer_id": reviewer_id,
            "reviewed_at": time.time(),
            "root_cause_analysis": root_cause_analysis,
            "status": "REVIEWED",
        }
        incident["status"] = "ACCOUNTABILITY_REVIEWED"

        # Log to audit trail
        if self.cfg.audit_log_enabled:
            self._audit_log("ACCOUNTABILITY_REVIEW", incident)

        return {
            "status": "REVIEWED",
            "incident_id": incident_id,
            "reviewer_id": reviewer_id,
            "note": (
                "Accountability review completed. "
                "Root cause analysis recorded. "
                "Proceed to transformation phase."
            ),
        }

    # ── Phase 5: TRANSFORMATION ───────────────────────────────────────────

    def initiate_transformation(
        self,
        incident_id: str,
        recovery_plan: str,
        metric_reevaluation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Initiate transformation/recovery (Phase 5 of response).

        From Repentance §11.3: Transformation comes LAST.
        From paper5 §1.1: "apparent disorder is very often a sign that
        the wrong statistic was sought" → metric re-evaluation required.

        Args:
            incident_id: ID of incident to recover from
            recovery_plan: Description of recovery plan
            metric_reevaluation: Optional metric re-evaluation notes

        Returns:
            dict with transformation results
        """
        # Find incident
        incident = None
        for inc in self.incidents:
            if inc["incident_id"] == incident_id:
                incident = inc
                break

        if incident is None:
            return {"status": "error", "error": "Incident not found"}

        # Update incident
        incident["transformation"] = {
            "recovery_plan": recovery_plan,
            "metric_reevaluation": metric_reevaluation,
            "initiated_at": time.time(),
            "status": "TRANSFORMATION_INITIATED",
        }
        incident["status"] = "TRANSFORMATION_INITIATED"

        # Log to audit trail
        if self.cfg.audit_log_enabled:
            self._audit_log("TRANSFORMATION_INITIATED", incident)

        return {
            "status": "TRANSFORMATION_INITIATED",
            "incident_id": incident_id,
            "note": (
                "Transformation/recovery initiated. "
                "Remember: 'apparent disorder is very often a sign that "
                "the wrong statistic was sought' (paper5 §1.1). "
                "Re-evaluate quality metrics before declaring recovery."
            ),
        }

    def complete_recovery(self, incident_id: str) -> Dict[str, Any]:
        """Mark incident as recovered."""
        incident = None
        for inc in self.incidents:
            if inc["incident_id"] == incident_id:
                incident = inc
                break

        if incident is None:
            return {"status": "error", "error": "Incident not found"}

        incident["status"] = "RECOVERED"
        incident["recovered_at"] = time.time()

        # Reset active incident if this was it
        if self.active_incident and self.active_incident["incident_id"] == incident_id:
            self.active_incident = None

        # Reset energy budget
        self._energy_spent = 0.0

        if self.cfg.audit_log_enabled:
            self._audit_log("RECOVERY_COMPLETED", incident)

        return {
            "status": "RECOVERED",
            "incident_id": incident_id,
            "duration_s": incident.get("recovered_at", time.time()) - incident.get("initiated_at", 0),
            "note": "Incident recovered. Energy budget reset.",
        }

    # ── Runbook Execution ─────────────────────────────────────────────────

    def execute_runbook(
        self,
        quality_history: List[float],
        operator_id: str,
    ) -> Dict[str, Any]:
        """
        Execute the full incident response runbook.

        Phases:
          1. DETECTION — Hill estimator, occupation fraction, CUSUM
          2. CLASSIFICATION — Severity level, regime type
          3. PROTECTION — Emergency isolation, quality gate activation
          4. ACCOUNTABILITY — Audit trail, root cause analysis
          5. TRANSFORMATION — Recovery, metric re-evaluation

        Args:
            quality_history: Quality scores over time
            operator_id: ID of operator executing runbook

        Returns:
            dict with full runbook execution results
        """
        results = {
            "runbook_id": str(uuid.uuid4()),
            "executed_at": time.time(),
            "operator_id": operator_id,
            "phases": {},
        }

        # Phase 1: DETECTION
        detection = self.detect_degenerate_regime(quality_history)
        results["phases"]["detection"] = detection

        if not detection.get("detected"):
            results["status"] = "NO_INCIDENT"
            results["note"] = "No degenerate regime detected. Normal operation."
            return results

        # Phase 2: CLASSIFICATION
        classification = self.classify_incident(detection, quality_history)
        results["phases"]["classification"] = classification

        # Phase 3: PROTECTION
        protection = self.initiate_protection(classification, operator_id)
        results["phases"]["protection"] = protection

        if protection.get("status") == "ENERGY_BUDGET_EXCEEDED":
            results["status"] = "ENERGY_BUDGET_EXCEEDED"
            results["note"] = protection.get("note")
            return results

        # Phase 4: ACCOUNTABILITY (requires human input)
        results["phases"]["accountability"] = {
            "status": "PENDING_HUMAN_REVIEW",
            "note": (
                "Accountability review requires human input. "
                "Call conduct_accountability_review() with root cause analysis."
            ),
        }

        # Phase 5: TRANSFORMATION (requires human input)
        results["phases"]["transformation"] = {
            "status": "PENDING_HUMAN_REVIEW",
            "note": (
                "Transformation requires human input. "
                "Call initiate_transformation() with recovery plan. "
                "Remember paper5: 'apparent disorder is very often a sign "
                "that the wrong statistic was sought'."
            ),
        }

        results["status"] = "PROTECTION_INITIATED"
        results["incident_id"] = protection.get("incident_id")
        results["note"] = (
            f"Degenerate regime incident detected and protection initiated. "
            f"Severity: {classification.get('severity_name')}. "
            f"Actions: {protection.get('actions')}. "
            f"Human review required for accountability and transformation phases."
        )

        return results

    # ── Audit Logging ─────────────────────────────────────────────────────

    def _audit_log(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an audit event."""
        log_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data,
        }

        # Write to file if configured
        if self.cfg.audit_log_path:
            try:
                with open(self.cfg.audit_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write audit log: {e}")

        logger.info(f"AUDIT: {event_type} — {data.get('incident_id', 'N/A')}")

    def get_incident_history(self) -> List[Dict[str, Any]]:
        """Get all incident history."""
        return self.incidents.copy()

    def get_active_incident(self) -> Optional[Dict[str, Any]]:
        """Get currently active incident."""
        return self.active_incident

    def get_status(self) -> Dict[str, Any]:
        """Get runbook status."""
        return {
            "total_incidents": len(self.incidents),
            "active_incident": self.active_incident is not None,
            "energy_spent": self._energy_spent,
            "energy_budget": self.cfg.max_response_energy,
            "config": {
                "degenerate_theta_threshold": self.cfg.degenerate_theta_threshold,
                "polarization_threshold": self.cfg.polarization_threshold,
                "cusum_threshold": self.cfg.cusum_threshold,
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class SESICloudSecuritySuite:
    """
    [D] Complete cloud security suite orchestrator.

    Integrates:
      • Terraform multi-cloud KMS generation
      • FIDO MDS client
      • Degenerate regime incident runbook
    """

    def __init__(self):
        self.terraform = TerraformMultiCloudKMSGenerator()
        self.fido_mds = FIDOMetadataServiceClient()
        self.runbook = DegenerateRegimeIncidentRunbook()

    def generate_all_terraform(self) -> Dict[str, Dict[str, str]]:
        """Generate all Terraform modules."""
        return self.terraform.generate_all()

    def check_fido_authenticator(self, aaguid: str) -> Dict[str, Any]:
        """Check FIDO authenticator status."""
        return self.fido_mds.check_attestation_status(aaguid)

    def respond_to_degenerate_regime(
        self,
        quality_history: List[float],
        operator_id: str,
    ) -> Dict[str, Any]:
        """Execute incident response runbook."""
        return self.runbook.execute_runbook(quality_history, operator_id)

    def status_report(self) -> Dict[str, Any]:
        """Get full system status."""
        return {
            "terraform": {
                "output_dir": str(self.terraform.output_dir),
            },
            "fido_mds": self.fido_mds.get_status(),
            "runbook": self.runbook.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════

def smoke_test():
    print("=" * 72)
    print(" SESI Terraform × FIDO MDS × Incident Runbook — Smoke Test")
    print("=" * 72)
    print(f"  PyJWT available      : {HAS_PYJWT}")
    print(f"  requests available   : {HAS_REQUESTS}")
    print(f"  PyYAML available     : {HAS_YAML}")
    print(f"  Previous modules     : {HAS_PREV_MODULES}")
    print(f"  AGIONE available     : {HAS_AGIONE}")

    suite = SESICloudSecuritySuite()

    # ── Test 1: Terraform Generation ──────────────────────────────────────
    print("\n  ── Terraform Multi-Cloud KMS Generation ──")
    terraform_files = suite.generate_all_terraform()
    for provider, files in terraform_files.items():
        print(f"    {provider:10s}: {len(files)} files generated")

    # ── Test 2: FIDO MDS Client ───────────────────────────────────────────
    print("\n  ── FIDO MDS Client ──")
    mds_status = suite.fido_mds.get_status()
    print(f"    Entries loaded     : {mds_status['entries_loaded']}")

    # Test with known AAGUIDs (simulated)
    test_aaguids = [
        "08987058-cadc-4b81-b6e1-30de50dcbe96",  # Windows Hello
        "6028b017-b1d4-4c02-b4b3-afcdafc96bb2",  # Touch ID
        "adce0a90-1e59-4ecc-8c0b-6e020557d7bd",  # YubiKey 5
    ]
    for aaguid in test_aaguids:
        result = suite.check_fido_authenticator(aaguid)
        status = result.get("status", "unknown")
        print(f"    AAGUID {aaguid[:8]}...: {status}")

    # ── Test 3: Degenerate Regime Detection ───────────────────────────────
    print("\n  ── Degenerate Regime Detection ──")

    # Simulate degenerate quality history (polarized)
    np.random.seed(42)
    degenerate_history = []
    for i in range(50):
        if i % 2 == 0:
            degenerate_history.append(0.95 + np.random.uniform(-0.02, 0.02))
        else:
            degenerate_history.append(0.05 + np.random.uniform(-0.02, 0.02))

    detection = suite.runbook.detect_degenerate_regime(degenerate_history)
    print(f"    Detected           : {detection['detected']}")
    print(f"    Regime             : {detection['regime']}")
    print(f"    θ̂ (Hill)           : {detection['theta_hat']:.4f}")
    print(f"    Occupation fraction: {detection['occupation_fraction']:.3f}")
    print(f"    Polarized          : {detection['polarized']}")

    # ── Test 4: Incident Classification ───────────────────────────────────
    print("\n  ── Incident Classification ──")
    classification = suite.runbook.classify_incident(detection, degenerate_history)
    print(f"    Severity           : {classification['severity_name']}")
    print(f"    Classification     : {classification['classification']}")
    print(f"    Recommended        : {classification['recommended_response']}")

    # ── Test 5: Full Runbook Execution ────────────────────────────────────
    print("\n  ── Full Runbook Execution ──")
    runbook_result = suite.respond_to_degenerate_regime(
        degenerate_history,
        operator_id="operator_001",
    )
    print(f"    Status             : {runbook_result['status']}")
    print(f"    Incident ID        : {runbook_result.get('incident_id', 'N/A')}")
    if "phases" in runbook_result:
        for phase, result in runbook_result["phases"].items():
            status = result.get("status", result.get("recommended_action", "N/A"))
            print(f"    Phase {phase:15s}: {status}")

    # ── Test 6: Normal Quality History ────────────────────────────────────
    print("\n  ── Normal Quality History (No Incident) ──")
    normal_history = list(np.random.uniform(0.6, 0.9, 50))
    normal_result = suite.respond_to_degenerate_regime(
        normal_history,
        operator_id="operator_001",
    )
    print(f"    Status             : {normal_result['status']}")
    print(f"    Note               : {normal_result.get('note', 'N/A')[:60]}...")

    # ── Full Status Report ────────────────────────────────────────────────
    print("\n  ── Full Status Report ──")
    status = suite.status_report()
    print(f"    Terraform output   : {status['terraform']['output_dir']}")
    print(f"    FIDO MDS entries   : {status['fido_mds']['entries_loaded']}")
    print(f"    Runbook incidents  : {status['runbook']['total_incidents']}")
    print(f"    Energy spent       : {status['runbook']['energy_spent']}/{status['runbook']['energy_budget']}")

    print("\n" + "=" * 72)
    print("  HONEST SCOPE")
    print("  • Terraform templates are CONFIGURATION GENERATORS, not security")
    print("  • China cloud compliance (等保, 密码法) requires separate certification")
    print("  • FIDO MDS verification checks PROVENANCE, not user identity")
    print("  • Incident runbook is a STRUCTURED GUIDE, not a substitute for")
    print("    human judgment")
    print("  • All thresholds are FREE DESIGN PARAMETERS")
    print("  • Degenerate regime detection is STATISTICAL, not deterministic")
    print("  • Energy budget limits response actions (No-Zeno Thm 4.2)")
    print("=" * 72)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    smoke_test()
