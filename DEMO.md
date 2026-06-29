# Сценарий итоговой демонстрации

1. Показать GitHub Actions: lint, tests, два Docker-образа и update `gitops`.
2. Показать Argo CD Application в состоянии `Synced / Healthy`.
3. Открыть Streamlit и выполнить inference без label, затем с `true_label`.
4. Показать запись в Predictions и соответствующие Prometheus/Grafana метрики.
5. Загрузить не менее 20 размеченных current samples или использовать подготовленные.
6. Запустить drift report и открыть timestamped HTML report.
7. Показать постоянный alert и созданный автоматический retrain Job.
8. Открыть MLflow run: параметры, метрики, confusion matrix и model version.
9. Показать alias `champion` и изменение `model_version` в `/health` после polling.
10. Запустить ручной retrain из UI и показать историю Jobs.
11. Повторно синхронизировать Argo CD и показать, что bootstrap hook завершается
    без immutable Job ошибки и без смены существующего champion.

Быстрая проверка:

```bash
kubectl get application -n argocd mlops-eyes
kubectl get pods,jobs,cronjobs -n mlops-eyes
curl http://localhost:8000/ready
curl http://localhost:8000/metrics
dvc status train
```
