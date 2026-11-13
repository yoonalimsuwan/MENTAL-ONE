#!/usr/bin/env python3
"""
=============================================================================
CLINICAL VALIDATION FRAMEWORK + EHR/FHIR INTEGRATION SCAFFOLD
=============================================================================
Developers : PAI AND Yoon A Limsuwan / MSPS NETWORK
             My Soul Move By Power of Holy Spirit ;
             What MSPS NETWORK SEE, Lord Buddha Knows.
License    : MIT | Year: 2026

╔═══════════════════════════════════════════════════════════════════════╗
║  ⚠️  CRITICAL — READ BEFORE ANYTHING ELSE                           ║
║                                                                     ║
║  THIS SOFTWARE IS A RESEARCH SCAFFOLD. IT IS NOT A MEDICAL DEVICE. ║
║  IT IS NOT VALIDATED FOR CLINICAL USE. IT MUST NOT BE USED TO      ║
║  MAKE, CONFIRM, OR RULE OUT ANY PSYCHIATRIC OR NEUROLOGICAL        ║
║  DIAGNOSIS. IT MUST NOT BE USED TO GUIDE TREATMENT DECISIONS.      ║
║                                                                     ║
║  Any clinical deployment requires:                                 ║
║    1. IRB/Ethics Committee approval                                 ║
║    2. Informed patient consent                                      ║
║    3. Multi-site validation against DSM-5/ICD-11 gold standard      ║
║    4. Regulatory approval (FDA 510(k)/De Novo, CE MDR, Thai FDA)   ║
║    5. Oversight by qualified psychiatrists at every step            ║
║                                                                     ║
║  Using this software for clinical decision-making without all of   ║
║  the above is dangerous and unethical.                             ║
╚═══════════════════════════════════════════════════════════════════════╝

WHAT THIS FILE PROVIDES (research/technical only)
  §1  FHIR R4 resource models for EEG/MEG/fMRI + psychiatric data
  §2  EHR integration layer (FHIR REST client scaffold)
  §3  De-identification (HIPAA Safe Harbor + GDPR)
  §4  Clinical validation study design scaffold
  §5  Audit logging + consent tracking
  §6  Research-only output gating

WHAT THIS FILE IS NOT
  ❌ Not a diagnostic tool
  ❌ Not validated on any patient data
  ❌ Not approved by any regulatory body
  ❌ Not a substitute for clinical judgment
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("clinical_ehr")

VERSION = "0.1.0-research"
ATTRIBUTION = ("PAI AND Yoon A Limsuwan / MSPS NETWORK — "
               "My Soul Move By Power of Holy Spirit ; "
               "What MSPS NETWORK SEE, Lord Buddha Knows.")

# =============================================================================
# §0. RESEARCH-ONLY GUARD
# =============================================================================

RESEARCH_ONLY_BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║  ⚠️  RESEARCH USE ONLY — NOT A MEDICAL DEVICE                       ║
║  NOT VALIDATED FOR CLINICAL DIAGNOSIS                                ║
║  ALL OUTPUTS ARE RESEARCH HYPOTHESES, NOT DIAGNOSES                  ║
║  CLINICAL DECISIONS MUST BE MADE BY QUALIFIED PSYCHIATRISTS          ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

def research_guard():
    """Print the research-only banner. Call at every entry point."""
    print(RESEARCH_ONLY_BANNER)


# =============================================================================
# §1. FHIR R4 RESOURCE MODELS
# =============================================================================
# HL7 FHIR R4 (https://hl7.org/fhir/R4/)
# These are minimal, research-oriented FHIR resources for psychiatric
# signal data. They are NOT complete FHIR implementations.

class ResourceType(str, Enum):
    PATIENT = "Patient"
    OBSERVATION = "Observation"
    DIAGNOSTIC_REPORT = "DiagnosticReport"
    CONDITION = "Condition"
    QUESTIONNAIRE_RESPONSE = "QuestionnaireResponse"
    SERVICE_REQUEST = "ServiceRequest"
    DOCUMENT_REFERENCE = "DocumentReference"
    CONSENT = "Consent"
    AUDIT_EVENT = "AuditEvent"


@dataclass
class FHIRIdentifier:
    system: str
    value: str
    use: str = "official"

    def to_dict(self):
        return {"system": self.system, "value": self.value, "use": self.use}


@dataclass
class FHIRCoding:
    system: str
    code: str
    display: str = ""

    def to_dict(self):
        return {"system": self.system, "code": self.code, "display": self.display}


@dataclass
class FHIRPatient:
    """
    FHIR Patient resource.
    For RESEARCH: all identifiers must be de-identified before storage.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    identifier: List[FHIRIdentifier] = field(default_factory=list)
    gender: str = "unknown"
    birthDate: str = ""
    consent_given: bool = False
    consent_date: str = ""
    research_id: str = ""  # de-identified research identifier

    def to_dict(self):
        return {
            "resourceType": ResourceType.PATIENT.value,
            "id": self.id,
            "identifier": [i.to_dict() for i in self.identifier],
            "gender": self.gender,
            "birthDate": self.birthDate,
            "extension": [
                {"url": "research-consent",
                 "valueBoolean": self.consent_given},
                {"url": "research-id",
                 "valueString": self.research_id},
            ],
        }


@dataclass
class FHIRObservation:
    """
    FHIR Observation for EEG/MEG/fMRI signal data or derived features.
    LOINC codes used where applicable.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "final"
    category: List[FHIRCoding] = field(default_factory=list)
    code: FHIRCoding = field(default_factory=lambda: FHIRCoding(
        system="http://loinc.org", code="UNKNOWN", display="Research observation"))
    subject: str = ""
    effectiveDateTime: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    valueQuantity: Optional[Dict[str, Any]] = None
    component: List[Dict[str, Any]] = field(default_factory=list)
    interpretation: List[FHIRCoding] = field(default_factory=list)
    note: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self):
        d = {
            "resourceType": ResourceType.OBSERVATION.value,
            "id": self.id,
            "status": self.status,
            "category": [c.to_dict() for c in self.category],
            "code": self.code.to_dict(),
            "subject": {"reference": f"Patient/{self.subject}"},
            "effectiveDateTime": self.effectiveDateTime,
        }
        if self.valueQuantity:
            d["valueQuantity"] = self.valueQuantity
        if self.component:
            d["component"] = self.component
        if self.interpretation:
            d["interpretation"] = [c.to_dict() for c in self.interpretation]
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class FHIRDiagnosticReport:
    """
    FHIR DiagnosticReport for analysis results.
    CRITICAL: status must be 'preliminary' or 'research' — never 'final'
    for unvalidated research output.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "preliminary"  # NEVER 'final' for research output
    category: List[FHIRCoding] = field(default_factory=lambda: [
        FHIRCoding(system="http://loinc.org", code="LP29684-5",
                   display="Radiology")  # placeholder
    ])
    code: FHIRCoding = field(default_factory=lambda: FHIRCoding(
        system="http://loinc.org", code="UNKNOWN",
        display="Research analysis — NOT a clinical report"))
    subject: str = ""
    effectiveDateTime: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    conclusion: str = "RESEARCH OUTPUT — NOT A CLINICAL DIAGNOSIS"
    presentedForm: List[Dict] = field(default_factory=list)
    result: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "resourceType": ResourceType.DIAGNOSTIC_REPORT.value,
            "id": self.id,
            "status": self.status,
            "category": [c.to_dict() for c in self.category],
            "code": self.code.to_dict(),
            "subject": {"reference": f"Patient/{self.subject}"},
            "effectiveDateTime": self.effectiveDateTime,
            "conclusion": self.conclusion,
            "result": [{"reference": f"Observation/{r}"} for r in self.result],
        }


@dataclass
class FHIRCondition:
    """
    FHIR Condition for research hypothesis only.
    verificationStatus MUST be 'unconfirmed' for research output.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    clinicalStatus: FHIRCoding = field(default_factory=lambda: FHIRCoding(
        system="http://terminology.hl7.org/CodeSystem/condition-clinical",
        code="active", display="Active"))
    verificationStatus: FHIRCoding = field(default_factory=lambda: FHIRCoding(
        system="http://terminology.hl7.org/CodeSystem/condition-ver-status",
        code="unconfirmed",  # CRITICAL: never 'confirmed' for research
        display="Unconfirmed — RESEARCH HYPOTHESIS ONLY"))
    code: FHIRCoding = field(default_factory=lambda: FHIRCoding(
        system="http://snomed.info/sct", code="UNKNOWN",
        display="Research hypothesis"))
    subject: str = ""
    note: List[Dict] = field(default_factory=lambda: [
        {"text": "RESEARCH HYPOTHESIS — NOT A CLINICAL DIAGNOSIS. "
                 "Must be evaluated by a qualified psychiatrist."}
    ])

    def to_dict(self):
        return {
            "resourceType": ResourceType.CONDITION.value,
            "id": self.id,
            "clinicalStatus": self.clinicalStatus.to_dict(),
            "verificationStatus": self.verificationStatus.to_dict(),
            "code": self.code.to_dict(),
            "subject": {"reference": f"Patient/{self.subject}"},
            "note": self.note,
        }


@dataclass
class FHIRQuestionnaireResponse:
    """
    FHIR QuestionnaireResponse for standardized psychiatric instruments.
    Supported: PHQ-9, GAD-7, PCL-5, YMRS, PANSS, HAMD-17.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "completed"
    questionnaire: str = ""
    subject: str = ""
    authored: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    item: List[Dict[str, Any]] = field(default_factory=list)

    # Standard questionnaire canonical URLs
    QUESTIONNAIRES = {
        "PHQ-9": "http://hl7.org/fhir/Questionnaire/phq-9",
        "GAD-7": "http://hl7.org/fhir/Questionnaire/gad-7",
        "PCL-5": "http://hl7.org/fhir/Questionnaire/pcl-5",
        "YMRS": "http://hl7.org/fhir/Questionnaire/ymrs",
        "PANSS": "http://hl7.org/fhir/Questionnaire/panss",
        "HAMD-17": "http://hl7.org/fhir/Questionnaire/hamd-17",
    }

    def to_dict(self):
        return {
            "resourceType": ResourceType.QUESTIONNAIRE_RESPONSE.value,
            "id": self.id,
            "status": self.status,
            "questionnaire": self.questionnaire,
            "subject": {"reference": f"Patient/{self.subject}"},
            "authored": self.authored,
            "item": self.item,
        }


@dataclass
class FHIRConsent:
    """
    FHIR Consent resource for research participation.
    REQUIRED before any patient data is processed.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "active"
    scope: FHIRCoding = field(default_factory=lambda: FHIRCoding(
        system="http://terminology.hl7.org/CodeSystem/consentscope",
        code="research", display="Research"))
    category: List[FHIRCoding] = field(default_factory=lambda: [
        FHIRCoding(system="http://loinc.org", code="59284-0",
                   display="Consent Document")
    ])
    patient: str = ""
    dateTime: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provision: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False

    def to_dict(self):
        return {
            "resourceType": ResourceType.CONSENT.value,
            "id": self.id,
            "status": self.status,
            "scope": self.scope.to_dict(),
            "category": [c.to_dict() for c in self.category],
            "patient": {"reference": f"Patient/{self.patient}"},
            "dateTime": self.dateTime,
            "provision": self.provision,
            "verified": self.verified,
        }


# =============================================================================
# §2. EEG/MEG/fMRI → FHIR OBSERVATION TRANSFORM
# =============================================================================

# LOINC codes for neurophysiology observations
LOINC_EEG = "11526-1"      # EEG
LOINC_MEG = "LP29251-4"    # MEG (placeholder, verify)
LOINC_MRI_BRAIN = "LP29695-5"  # MRI Brain

# SNOMED CT codes for psychiatric conditions (research use)
SNOMED_MDD = "370143000"           # Major depressive disorder
SNOMED_BIPOLAR = "13746004"        # Bipolar disorder
SNOMED_SCHIZOPHRENIA = "58214004"  # Schizophrenia
SNOMED_PTS_D = "443853005"         # PTSD
SNOMED_ANXIETY = "48694002"        # Anxiety disorder
SNOMED_OCD = "191736004"           # OCD
SNOMED_PANIC = "197480006"         # Panic disorder


def eeg_to_observation(
    patient_id: str,
    eeg_data: Any,  # torch.Tensor or numpy array
    sampling_rate: float = 256.0,
    channels: Optional[List[str]] = None,
    duration_seconds: Optional[float] = None,
) -> FHIRObservation:
    """
    Convert raw EEG tensor to FHIR Observation.

    CRITICAL: eeg_data must be DE-IDENTIFIED before this function is called.
    No patient identifiers should be embedded in the signal data.
    """
    import numpy as np
    if hasattr(eeg_data, 'cpu'):
        data_np = eeg_data.detach().cpu().numpy()
    else:
        data_np = np.asarray(eeg_data)

    if channels is None:
        channels = [f"ch_{i}" for i in range(data_np.shape[0])]

    components = []
    for i, ch in enumerate(channels):
        if i < data_np.shape[0]:
            ch_data = data_np[i]
            components.append({
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": LOINC_EEG,
                        "display": f"EEG channel {ch}",
                    }]
                },
                "valueQuantity": {
                    "value": float(np.mean(ch_data)),
                    "unit": "uV",
                    "system": "http://unitsofmeasure.org",
                    "code": "uV",
                },
                "extension": [{
                    "url": "peak-to-peak-amplitude",
                    "valueQuantity": {
                        "value": float(np.ptp(ch_data)),
                        "unit": "uV",
                    }
                }],
            })

    return FHIRObservation(
        status="final",
        category=[FHIRCoding(
            system="http://terminology.hl7.org/CodeSystem/observation-category",
            code="imaging", display="Imaging")],
        code=FHIRCoding(system="http://loinc.org", code=LOINC_EEG,
                        display="Electroencephalogram (EEG)"),
        subject=patient_id,
        component=components,
        note=[{"text": "RESEARCH DATA — de-identified. "
                       "Not for clinical interpretation."}],
    )


def questionnaire_to_fhir(
    patient_id: str,
    instrument: str,
    scores: Dict[str, float],
    total_score: float,
) -> FHIRQuestionnaireResponse:
    """
    Convert questionnaire scores to FHIR QuestionnaireResponse.
    instrument: one of PHQ-9, GAD-7, PCL-5, YMRS, PANSS, HAMD-17
    """
    q_url = FHIRQuestionnaireResponse.QUESTIONNAIRES.get(
        instrument, f"http://example.org/Questionnaire/{instrument}")

    items = []
    for q_id, score in scores.items():
        items.append({
            "linkId": str(q_id),
            "answer": [{
                "valueDecimal": float(score),
            }],
        })

    return FHIRQuestionnaireResponse(
        questionnaire=q_url,
        subject=patient_id,
        item=items,
    )


def analysis_to_diagnostic_report(
    patient_id: str,
    observations: List[str],
    hypothesis_category: str,
    hypothesis_code: str,
    hypothesis_display: str,
    confidence: float,
) -> Tuple[FHIRDiagnosticReport, FHIRCondition]:
    """
    Create FHIR DiagnosticReport + Condition for a RESEARCH HYPOTHESIS.

    CRITICAL:
    - DiagnosticReport.status = 'preliminary' (NEVER 'final')
    - Condition.verificationStatus = 'unconfirmed' (NEVER 'confirmed')
    - All conclusions are marked as research hypotheses
    - A qualified psychiatrist MUST review before any clinical action
    """
    report = FHIRDiagnosticReport(
        status="preliminary",  # CRITICAL: never 'final' for research
        category=[FHIRCoding(
            system="http://terminology.hl7.org/CodeSystem/v2-0074",
            code="NB", display="Neurophysiology")],
        code=FHIRCoding(
            system="http://loinc.org",
            code="UNKNOWN",
            display="RESEARCH: Computational psychiatry analysis"),
        subject=patient_id,
        conclusion=(
            f"RESEARCH HYPOTHESIS (confidence: {confidence:.2f}): "
            f"{hypothesis_display}. "
            "THIS IS NOT A CLINICAL DIAGNOSIS. "
            "MUST BE EVALUATED BY A QUALIFIED PSYCHIATRIST. "
            "This output was generated by a research algorithm that has "
            "NOT been validated for clinical use."
        ),
        result=observations,
    )

    condition = FHIRCondition(
        verificationStatus=FHIRCoding(
            system="http://terminology.hl7.org/CodeSystem/condition-ver-status",
            code="unconfirmed",
            display="Unconfirmed — RESEARCH HYPOTHESIS ONLY"),
        code=FHIRCoding(
            system="http://snomed.info/sct",
            code=hypothesis_code,
            display=hypothesis_display),
        subject=patient_id,
        note=[{"text": (
            f"RESEARCH HYPOTHESIS — confidence {confidence:.2f}. "
            "NOT A CLINICAL DIAGNOSIS. This hypothesis was generated by "
            "an unvalidated research algorithm. A qualified psychiatrist "
            "must independently evaluate the patient before any diagnosis "
            "is made or treatment is initiated."
        )}],
    )

    return report, condition


# =============================================================================
# §3. DE-IDENTIFICATION (HIPAA Safe Harbor + GDPR)
# =============================================================================

# HIPAA Safe Harbor identifiers that must be removed
HIPAA_SAFE_HARBOR = [
    "name", "address", "city", "state", "zip", "phone", "fax",
    "email", "ssn", "medical_record_number", "health_plan_number",
    "account_number", "certificate_number", "license_number",
    "vehicle_identifier", "device_identifier", "url", "ip_address",
    "biometric_identifier", "photograph", "any_other_unique_identifier",
]


def deidentify_patient(
    patient: FHIRPatient,
    salt: str = "RESEARCH_SALT_CHANGE_ME",
) -> FHIRPatient:
    """
    De-identify a FHIR Patient resource using HIPAA Safe Harbor method.
    Replaces identifiers with salted hashes.
    """
    deidentified = FHIRPatient(
        id=patient.id,
        gender=patient.gender,
        # Remove birthDate — use age range instead
        birthDate="",
        consent_given=patient.consent_given,
        consent_date=patient.consent_date,
    )

    # Create research ID from salted hash
    research_id = hashlib.sha256(
        f"{salt}:{patient.id}".encode()
    ).hexdigest()[:16]
    deidentified.research_id = research_id

    # Remove all direct identifiers
    deidentified.identifier = [
        FHIRIdentifier(
            system="urn:research-id",
            value=research_id,
            use="usual",
        )
    ]

    return deidentified


def scrub_phi(text: str) -> str:
    """
    Remove potential PHI from free text.
    This is a basic regex-based scrubber — NOT a substitute for
    proper NLP-based de-identification in production.
    """
    import re
    # Remove potential MRNs (patterns vary by institution)
    text = re.sub(r'\b[A-Z]{2,3}-?\d{6,10}\b', '[MRN_REMOVED]', text)
    # Remove potential phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REMOVED]', text)
    # Remove potential SSNs
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REMOVED]', text)
    # Remove potential email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  '[EMAIL_REMOVED]', text)
    return text


# =============================================================================
# §4. EHR INTEGRATION LAYER (FHIR REST CLIENT SCAFFOLD)
# =============================================================================

class FHIRClient:
    """
    Minimal FHIR R4 REST client scaffold.

    In production, use a proper FHIR client library:
    - Python: fhirclient (https://github.com/smart-on-fhir/client-py)
    - Or vendor-specific EHR APIs (Epic, Cerner/Oracle Health, etc.)

    This scaffold demonstrates the integration pattern, NOT a complete
    FHIR implementation.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.audit_log: List[Dict[str, Any]] = []

    def _log_access(self, method: str, resource_type: str,
                    resource_id: str, user: str = "system"):
        """Audit log entry for every EHR access (HIPAA requirement)."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "resourceType": resource_type,
            "resourceId": resource_id,
            "user": user,
            "purpose": "RESEARCH",
        }
        self.audit_log.append(entry)
        log.info(f"EHR access: {method} {resource_type}/{resource_id}")

    def read(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        """
        Read a FHIR resource.

        In production, this would make an HTTP GET request:
        GET {base_url}/{resource_type}/{resource_id}

        Here we return a placeholder to demonstrate the pattern.
        """
        self._log_access("GET", resource_type, resource_id)
        # PRODUCTION: requests.get(f"{self.base_url}/{resource_type}/{resource_id}",
        #                          headers=self._headers())
        return {
            "resourceType": resource_type,
            "id": resource_id,
            "_note": "SCAFFOLD — implement actual HTTP call for production",
        }

    def create(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a FHIR resource.

        In production, this would make an HTTP POST request:
        POST {base_url}/{resource_type}
        """
        rt = resource.get("resourceType", "Unknown")
        rid = resource.get("id", str(uuid.uuid4()))
        self._log_access("POST", rt, rid)
        # PRODUCTION: requests.post(f"{self.base_url}/{rt}",
        #                           json=resource, headers=self._headers())
        return {"created": rt, "id": rid}

    def search(self, resource_type: str,
               params: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Search FHIR resources.

        In production, this would make an HTTP GET request:
        GET {base_url}/{resource_type}?param=value
        """
        self._log_access("SEARCH", resource_type, str(params))
        # PRODUCTION: requests.get(f"{self.base_url}/{resource_type}",
        #                          params=params, headers=self._headers())
        return []

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/fhir+json",
            "Accept": "application/fhir+json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class EHRIntegration:
    """
    High-level EHR integration for the clinical psychiatry analyzer.

    Workflow:
    1. Verify patient consent (FHIR Consent resource)
    2. Retrieve EEG/clinical data from EHR
    3. De-identify data
    4. Run analysis (research only)
    5. Store results as FHIR resources with research-only flags
    6. Log all access
    """

    def __init__(self, fhir_client: FHIRClient):
        self.client = fhir_client

    def verify_consent(self, patient_id: str) -> bool:
        """
        Verify that the patient has given research consent.
        MUST be called before any data access.
        """
        consents = self.client.search(
            ResourceType.CONSENT.value,
            {"patient": f"Patient/{patient_id}",
             "status": "active"}
        )
        # In production, check consent.provision for research scope
        has_consent = len(consents) > 0
        if not has_consent:
            log.warning(f"No active research consent for patient {patient_id}")
        return has_consent

    def retrieve_eeg(self, patient_id: str) -> Optional[FHIRObservation]:
        """Retrieve EEG observation from EHR."""
        if not self.verify_consent(patient_id):
            return None
        observations = self.client.search(
            ResourceType.OBSERVATION.value,
            {"patient": f"Patient/{patient_id}",
             "code": LOINC_EEG}
        )
        if observations:
            return observations[0]  # Return first match
        return None

    def store_research_result(
        self,
        patient_id: str,
        report: FHIRDiagnosticReport,
        condition: FHIRCondition,
    ) -> Dict[str, str]:
        """
        Store research results in EHR.
        All results are flagged as research-only.
        """
        if not self.verify_consent(patient_id):
            raise PermissionError("No research consent on file")

        report_id = self.client.create(report.to_dict())
        condition_id = self.client.create(condition.to_dict())

        return {
            "report_id": report_id.get("id", ""),
            "condition_id": condition_id.get("id", ""),
            "status": "STORED_AS_RESEARCH_HYPOTHESIS",
            "disclaimer": "NOT A CLINICAL DIAGNOSIS",
        }


# =============================================================================
# §5. CLINICAL VALIDATION STUDY DESIGN SCAFFOLD
# =============================================================================

@dataclass
class ValidationStudyConfig:
    """
    Configuration for a clinical validation study.

    This is a SCAFFOLD for study design, not an executable study.
    A real study requires:
    - IRB/Ethics Committee approval
    - Informed consent process
    - Data safety monitoring board
    - Pre-registered analysis plan
    - Multi-site recruitment
    - Qualified clinical raters
    """
    study_name: str = "Computational Psychiatry Signal Analysis Validation"
    protocol_version: str = "0.1"
    irb_approval: str = "PENDING"  # Must be obtained before any data collection
    primary_endpoint: str = "Diagnostic accuracy vs. structured clinical interview"
    gold_standard: str = "SCID-5 (Structured Clinical Interview for DSM-5)"
    target_conditions: List[str] = field(default_factory=lambda: [
        "Major Depressive Disorder",
        "Bipolar Disorder",
        "Schizophrenia Spectrum",
        "PTSD",
        "Panic Disorder",
    ])
    sample_size_per_condition: int = 200
    total_sample_size: int = 1000  # 200 per condition
    inclusion_criteria: List[str] = field(default_factory=lambda: [
        "Age 18-65",
        "Able to provide informed consent",
        "Meets DSM-5 criteria for target condition (confirmed by SCID-5)",
        "No current substance intoxication",
        "No unstable medical condition",
    ])
    exclusion_criteria: List[str] = field(default_factory=lambda: [
        "Active suicidal ideation with intent (requires immediate clinical care)",
        "Inability to consent",
        "Current ECT or TMS treatment",
        "Implanted metal devices incompatible with MRI/MEG",
    ])
    primary_outcome: str = "Sensitivity and specificity vs. SCID-5 diagnosis"
    secondary_outcomes: List[str] = field(default_factory=lambda: [
        "AUROC for each condition",
        "Positive predictive value",
        "Negative predictive value",
        "Agreement with treating psychiatrist's clinical diagnosis",
        "Test-retest reliability (ICC)",
    ])
    statistical_plan: str = (
        "Pre-registered analysis plan required. "
        "Primary analysis: sensitivity/specificity with 95% CI "
        "using exact binomial method. "
        "Multiplicity adjustment for multiple conditions. "
        "Subgroup analyses by age, sex, comorbidity."
    )
    data_safety_monitoring: str = (
        "Independent DSMB required. "
        "Stopping rules for adverse events. "
        "Regular safety reviews."
    )
    regulatory_pathway: str = (
        "FDA: SaMD (Software as a Medical Device) — "
        "likely Class II with 510(k) or De Novo pathway. "
        "EU: CE MDR Class IIa or IIb. "
        "Thailand: Thai FDA medical device registration."
    )

    def to_dict(self):
        return asdict(self)

    def summary(self) -> str:
        return f"""
╔═══════════════════════════════════════════════════════════════════════╗
║  CLINICAL VALIDATION STUDY DESIGN — SCAFFOLD ONLY                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Study: {self.study_name:<57} ║
║  Protocol: {self.protocol_version:<55} ║
║  IRB Status: {self.irb_approval:<53} ║
║                                                                     ║
║  Gold Standard: {self.gold_standard:<49} ║
║  Target N: {self.total_sample_size:<55} ║
║  Conditions: {len(self.target_conditions):<53} ║
║                                                                     ║
║  Primary Endpoint:                                                  ║
║    {self.primary_endpoint:<66} ║
║                                                                     ║
║  Regulatory Pathway:                                                ║
║    {self.regulatory_pathway[:66]:<66} ║
║                                                                     ║
║  ⚠️  THIS IS A STUDY DESIGN SCAFFOLD, NOT AN APPROVED PROTOCOL.    ║
║     IRB approval, informed consent, and regulatory submission       ║
║     are REQUIRED before any patient data is collected or analyzed.  ║
╚═══════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# §6. AUDIT LOGGING
# =============================================================================

class AuditLogger:
    """
    HIPAA-compliant audit logging.
    Every access to patient data must be logged.
    """

    def __init__(self, log_path: str = "audit_log.jsonl"):
        self.log_path = log_path
        self.entries: List[Dict[str, Any]] = []

    def log(self, action: str, resource_type: str, resource_id: str,
            user: str = "system", purpose: str = "RESEARCH",
            details: Optional[Dict] = None):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "resourceType": resource_type,
            "resourceId": resource_id,
            "user": user,
            "purpose": purpose,
            "details": details or {},
        }
        self.entries.append(entry)
        log.info(f"AUDIT: {action} {resource_type}/{resource_id} by {user}")

    def flush(self):
        """Write audit log to disk (append-only)."""
        import json
        with open(self.log_path, "a") as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + "\n")
        self.entries.clear()


# =============================================================================
# §7. SELF-TEST
# =============================================================================

def self_test() -> bool:
    research_guard()
    print("=" * 78)
    print(f"  CLINICAL EHR INTEGRATION v{VERSION}")
    print(f"  {ATTRIBUTION}")
    print("=" * 78)
    print("  ⚠️  RESEARCH SCAFFOLD — NOT FOR CLINICAL USE")
    print("=" * 78)
    ok = True

    def rep(name, passed, extra=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}  {extra}")

    # T1: FHIR Patient creation
    patient = FHIRPatient(
        gender="female",
        birthDate="1990-01-15",
        consent_given=True,
        consent_date="2026-01-01",
    )
    pd = patient.to_dict()
    rep("T1 FHIR Patient serializes", pd["resourceType"] == "Patient")

    # T2: De-identification
    deidentified = deidentify_patient(patient, salt="test_salt")
    rep("T2 De-identification removes birthDate", deidentified.birthDate == "")
    rep("T2 Research ID assigned", len(deidentified.research_id) > 0)
    rep("T2 Original ID not in research ID",
        patient.id not in deidentified.research_id)

    # T3: EEG Observation
    import numpy as np
    eeg_fake = np.random.randn(19, 256).astype(np.float32)
    obs = eeg_to_observation(
        patient_id=deidentified.research_id,
        eeg_data=eeg_fake,
        sampling_rate=256.0,
    )
    od = obs.to_dict()
    rep("T3 EEG Observation serializes", od["resourceType"] == "Observation")
    rep("T3 EEG has components", len(od.get("component", [])) > 0)
    rep("T3 EEG marked as research", "RESEARCH" in od["note"][0]["text"])

    # T4: Diagnostic Report + Condition
    report, condition = analysis_to_diagnostic_report(
        patient_id=deidentified.research_id,
        observations=[obs.id],
        hypothesis_category="MDD",
        hypothesis_code=SNOMED_MDD,
        hypothesis_display="Major Depressive Disorder",
        confidence=0.72,
    )
    rd = report.to_dict()
    cd = condition.to_dict()
    rep("T4 Report status is 'preliminary'", rd["status"] == "preliminary")
    rep("T4 Condition is 'unconfirmed'",
        cd["verificationStatus"]["code"] == "unconfirmed")
    rep("T4 Report conclusion has disclaimer",
        "NOT A CLINICAL DIAGNOSIS" in rd["conclusion"])
    rep("T4 Condition note has disclaimer",
        "NOT A CLINICAL DIAGNOSIS" in cd["note"][0]["text"])

    # T5: Questionnaire
    qr = questionnaire_to_fhir(
        patient_id=deidentified.research_id,
        instrument="PHQ-9",
        scores={i: 2.0 for i in range(9)},
        total_score=18.0,
    )
    qrd = qr.to_dict()
    rep("T5 Questionnaire serializes",
        qrd["resourceType"] == "QuestionnaireResponse")
    rep("T5 PHQ-9 canonical URL",
        "phq-9" in qrd["questionnaire"].lower())

    # T6: Consent
    consent = FHIRConsent(patient=deidentified.research_id, verified=True)
    consd = consent.to_dict()
    rep("T6 Consent serializes", consd["resourceType"] == "Consent")
    rep("T6 Consent is for research", consd["scope"]["code"] == "research")

    # T7: PHI scrubbing
    phi_text = "Patient John Doe, MRN AB-123456, phone 555-123-4567, " \
               "SSN 123-45-6789, email john@example.com"
    scrubbed = scrub_phi(phi_text)
    rep("T7 PHI scrubbed", "[MRN_REMOVED]" in scrubbed
        and "[PHONE_REMOVED]" in scrubbed
        and "[SSN_REMOVED]" in scrubbed
        and "[EMAIL_REMOVED]" in scrubbed)

    # T8: FHIR client scaffold
    client = FHIRClient(base_url="https://ehr.example.org/fhir")
    result = client.read("Patient", "12345")
    rep("T8 FHIR client read scaffold", result["resourceType"] == "Patient")
    rep("T8 Audit log populated", len(client.audit_log) > 0)

    # T9: Validation study config
    study = ValidationStudyConfig()
    rep("T9 Study config created", study.irb_approval == "PENDING")
    rep("T9 Gold standard is SCID-5", "SCID-5" in study.gold_standard)
    print("\n" + study.summary())

    # T10: Audit logger
    audit = AuditLogger(log_path="/tmp/test_audit.jsonl")
    audit.log("READ", "Patient", "test-123", purpose="RESEARCH")
    rep("T10 Audit log entry", len(audit.entries) == 1)

    print("=" * 78)
    print(f"  {'ALL TESTS PASSED' if ok else 'SOME TESTS FAILED'}")
    print("=" * 78)
    print("""
  WHAT THIS SCAFFOLD PROVIDES
    • FHIR R4 resource models for psychiatric signal data
    • De-identification (HIPAA Safe Harbor basic)
    • EHR integration pattern (FHIR REST client scaffold)
    • Clinical validation study design template
    • Audit logging
    • Research-only output gating

  WHAT IS REQUIRED BEFORE ANY CLINICAL USE
    1. IRB/Ethics Committee approval
    2. Informed consent from every participant
    3. Real EEG/MEG/fMRI data with confirmed diagnoses
    4. Multi-site validation against SCID-5 gold standard
    5. Regulatory submission (FDA/CE/Thai FDA)
    6. Independent clinical oversight at every step
    7. Proper NLP-based de-identification (not just regex)
    8. Security audit (penetration testing, encryption at rest/in transit)
    9. Data use agreements with all participating sites
    10. Publication of validation results for peer review

  THIS SOFTWARE MUST NEVER BE USED TO:
    ❌ Make a psychiatric diagnosis without a qualified psychiatrist
    ❑ Initiate, modify, or stop any treatment
    ❑ Deny care or insurance coverage
    ❑ Make legal determinations (competency, insanity defense, etc.)
    ❑ Replace clinical judgment
""")
    return ok


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Clinical EHR Integration Scaffold (RESEARCH ONLY)")
    parser.add_argument("--test", action="store_true",
                        help="Run self-tests")
    parser.add_argument("--study-design", action="store_true",
                        help="Print validation study design scaffold")
    args = parser.parse_args()

    if args.test:
        raise SystemExit(0 if self_test() else 1)

    if args.study_design:
        research_guard()
        study = ValidationStudyConfig()
        print(study.summary())
        return

    research_guard()
    parser.print_help()
    print(f"\n{ATTRIBUTION}")


if __name__ == "__main__":
    main()
