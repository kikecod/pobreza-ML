# CI/CD a AWS ECS (Elastic Container Service)

Este proyecto despliega la API en AWS ECS Fargate usando ECR como registro de imagenes.

## 1) Variables de GitHub (Repository Variables)

Configura estas variables en Settings > Secrets and variables > Actions > Variables:

- `AWS_REGION`: region AWS (ej. `us-east-1`)
- `ECR_REPOSITORY`: nombre del repositorio ECR (ej. `api-pobreza`)
- `ECS_CLUSTER`: nombre del cluster ECS
- `ECS_SERVICE`: nombre del servicio ECS
- `MODEL_AUC` (opcional): valor AUC del modelo cuando no exista `output/metrics.json`

## 2) Secret de GitHub

Configura este secret en Settings > Secrets and variables > Actions > Secrets:

- `AWS_ROLE_TO_ASSUME`: ARN del rol IAM para OIDC de GitHub Actions

## 3) Task Definition

Edita el archivo `.aws/ecs-task-definition.json`:

- Reemplaza `<ACCOUNT_ID>` por tu cuenta AWS
- Ajusta `executionRoleArn` y `taskRoleArn`
- Ajusta `awslogs-region`

El workflow actual inyecta automaticamente la imagen nueva en el contenedor `api-pobreza`.

## 4) Gate de calidad (AUC)

El despliegue se bloquea si `AUC < 0.84`.

Fuentes de AUC admitidas:

- `output/metrics.json` (generado por `main.py`)
- variable `MODEL_AUC`

## 5) Flujo

1. Push a `main`/`master`
2. CI instala dependencias y corre tests (si existen)
3. CI valida umbral AUC
4. Build Docker y push a ECR
5. Deploy del task definition en ECS Service
