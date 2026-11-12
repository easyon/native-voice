{{/*
Expand the name of the chart.
*/}}
{{- define "cosyvoice-runtime.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "cosyvoice-runtime.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "cosyvoice-runtime.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "cosyvoice-runtime.labels" -}}
helm.sh/chart: {{ include "cosyvoice-runtime.chart" . }}
{{ include "cosyvoice-runtime.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "cosyvoice-runtime.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cosyvoice-runtime.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "cosyvoice-runtime.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "cosyvoice-runtime.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "cosyvoice-runtime.secretName" -}}
{{- default (include "cosyvoice-runtime.fullname" .) .Values.runtime.auth.existingSecret }}
{{- end }}
