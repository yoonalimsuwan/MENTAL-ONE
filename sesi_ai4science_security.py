# =============================================================================
# SESI AI4Science Security Framework — Kubernetes Manifests
# =============================================================================
# Developer    : PAI, Yoon A Limsuwan / MSPS NETWORK
# MY SOUL MOVE BY POWER OF HOLY SPIRIT
# ORCID        : 0009-0008-2374-0788
# License      : MIT
# Year         : 2026
#
# HONEST SCOPE:
# • These are TEMPLATES requiring cluster-specific configuration
# • GPU resources are optional and must be explicitly enabled
# • TLS certificates must be provisioned via cert-manager
# • All secrets must be managed externally (Vault, AWS Secrets Manager)
# • Production deployment requires security audit
# =============================================================================

---
# Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: sesi-ai4science
  labels:
    app.kubernetes.io/name: sesi-ai4science
    app.kubernetes.io/part-of: sesi-ecosystem
    environment: production

---
# ConfigMap — Application configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: sesi-ai4science-config
  namespace: sesi-ai4science
data:
  config.yaml: |
    # AI4Science Security Framework Configuration
    environment: production
    logLevel: INFO

    # Output risk thresholds (paper §3.3)
    # HONEST: These are FREE DESIGN PARAMETERS
    riskThresholds:
      tau1_review: 0.4
      tau2_restrict: 0.7

    # Cumulative risk (paper §4.2)
    cumulativeRisk:
      lambda: 0.3
      historyWindowSize: 50
      combinationThreshold: 0.6

    # Multi-identity detection (paper §5.2)
    identityDetection:
      riskThreshold: 0.7
      coordinationThreshold: 0.8
      maxAccountsPerIdentity: 3

    # Data classification
    dataClassification:
      requireAuthForC2: true
      requireAuthForC3: true

    # Government oversight
    oversight:
      enableEmergencyIntervention: true
      requireLegalBasis: true
      maxInterventionDurationS: 86400

    # Scientific freedom safeguards
    freedomGuard:
      enableAppealMechanism: true
      requireProportionality: true
      independentReviewRequired: true

    # Model configuration
    model:
      useAMP: true
      useDDP: true
      useGradientCheckpointing: true
      device: cuda

---
# Secret — TLS certificates (managed by cert-manager)
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: sesi-ai4science-tls
  namespace: sesi-ai4science
spec:
  secretName: sesi-ai4science-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - ai4science.sesi.example.com
  privateKey:
    algorithm: RSA
    size: 2048

---
# Deployment — Main application
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sesi-ai4science-server
  namespace: sesi-ai4science
  labels:
    app.kubernetes.io/name: sesi-ai4science
    app.kubernetes.io/component: server
spec:
  replicas: 2
  selector:
    matchLabels:
      app.kubernetes.io/name: sesi-ai4science
      app.kubernetes.io/component: server
  template:
    metadata:
      labels:
        app.kubernetes.io/name: sesi-ai4science
        app.kubernetes.io/component: server
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: /metrics
    spec:
      serviceAccountName: sesi-ai4science
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault

      initContainers:
        # Wait for Redis dependency
        - name: wait-for-redis
          image: busybox:1.36
          command: ['sh', '-c', 'until nc -z sesi-ai4science-redis 6379; do echo waiting for redis; sleep 2; done;']

      containers:
        - name: ai4science-server
          image: ghcr.io/yoonalimsuwan/sesi-ai4science:latest
          imagePullPolicy: IfNotPresent

          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
            - name: websocket
              containerPort: 8765
              protocol: TCP
            - name: metrics
              containerPort: 9090
              protocol: TCP

          env:
            - name: SESI_ENVIRONMENT
              valueFrom:
                configMapKeyRef:
                  name: sesi-ai4science-config
                  key: config.yaml
            - name: SESI_LOG_LEVEL
              value: "INFO"
            - name: REDIS_HOST
              value: sesi-ai4science-redis
            - name: REDIS_PORT
              value: "6379"
            # GPU configuration (optional)
            - name: NVIDIA_VISIBLE_DEVICES
              value: "all"
            - name: CUDA_VISIBLE_DEVICES
              value: "0"

          resources:
            requests:
              cpu: "1000m"
              memory: "2Gi"
              # Uncomment for GPU support:
              # nvidia.com/gpu: 1
            limits:
              cpu: "4000m"
              memory: "8Gi"
              # Uncomment for GPU support:
              # nvidia.com/gpu: 1

          # Health checks
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 60
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3

          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 30
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3

          # Startup probe for slow model loading
          startupProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 10
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 60  # Allow up to 10 minutes for model loading

          volumeMounts:
            - name: config
              mountPath: /app/config
              readOnly: true
            - name: checkpoints
              mountPath: /app/checkpoints
            - name: tls-certs
              mountPath: /app/certs
              readOnly: true
            - name: tmp
              mountPath: /tmp

          # Security context
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop:
                - ALL
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            runAsUser: 1000

      volumes:
        - name: config
          configMap:
            name: sesi-ai4science-config
        - name: checkpoints
          persistentVolumeClaim:
            claimName: sesi-ai4science-checkpoints
        - name: tls-certs
          secret:
            secretName: sesi-ai4science-tls
            optional: true
        - name: tmp
          emptyDir: {}

      # Node affinity for GPU nodes
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              preference:
                matchExpressions:
                  - key: nvidia.com/gpu.present
                    operator: In
                    values:
                      - "true"
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app.kubernetes.io/name
                      operator: In
                      values:
                        - sesi-ai4science
                topologyKey: kubernetes.io/hostname

      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule

---
# Service
apiVersion: v1
kind: Service
metadata:
  name: sesi-ai4science-service
  namespace: sesi-ai4science
  labels:
    app.kubernetes.io/name: sesi-ai4science
spec:
  selector:
    app.kubernetes.io/name: sesi-ai4science
    app.kubernetes.io/component: server
  ports:
    - name: http
      port: 80
      targetPort: http
      protocol: TCP
    - name: websocket
      port: 8765
      targetPort: websocket
      protocol: TCP
    - name: metrics
      port: 9090
      targetPort: metrics
      protocol: TCP
  type: ClusterIP

---
# Ingress with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sesi-ai4science-ingress
  namespace: sesi-ai4science
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/websocket-services: "sesi-ai4science-service"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
spec:
  tls:
    - hosts:
        - ai4science.sesi.example.com
      secretName: sesi-ai4science-tls
  rules:
    - host: ai4science.sesi.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: sesi-ai4science-service
                port:
                  number: 80
          - path: /ws
            pathType: Prefix
            backend:
              service:
                name: sesi-ai4science-service
                port:
                  number: 8765

---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sesi-ai4science-hpa
  namespace: sesi-ai4science
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sesi-ai4science-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
        - type: Pods
          value: 4
          periodSeconds: 15
      selectPolicy: Max

---
# Pod Disruption Budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: sesi-ai4science-pdb
  namespace: sesi-ai4science
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: sesi-ai4science

---
# Persistent Volume Claim for checkpoints
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: sesi-ai4science-checkpoints
  namespace: sesi-ai4science
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard

---
# Service Account
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sesi-ai4science
  namespace: sesi-ai4science
  labels:
    app.kubernetes.io/name: sesi-ai4science

---
# Service Monitor for Prometheus
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sesi-ai4science-monitor
  namespace: sesi-ai4science
  labels:
    app.kubernetes.io/name: sesi-ai4science
    release: prometheus
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: sesi-ai4science
  endpoints:
    - port: metrics
      path: /metrics
      interval: 30s
      scrapeTimeout: 10s
  namespaceSelector:
    matchNames:
      - sesi-ai4science

---
# Network Policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: sesi-ai4science-netpol
  namespace: sesi-ai4science
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: sesi-ai4science
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8080
        - protocol: TCP
          port: 8765
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 9090
  egress:
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 443
        - protocol: TCP
          port: 6379
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
