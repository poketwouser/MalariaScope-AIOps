# MalariaScope: Automated Malaria Parasite Detection from Blood Smear Images

## Project Statement

### a) Problem Statement

Malaria is still one of the world's deadliest infectious diseases. The WHO World Malaria Report 2024 estimates about 263 million cases and 597,000 deaths in 2023, most of them in sub-Saharan Africa and South Asia. The standard way to diagnose it is to look at Giemsa-stained thin blood smears under a microscope and check red blood cells for Plasmodium parasites. This manual process is slow, tiring and depends heavily on the skill of the microscopist. Results vary between readers, and in the places where malaria is most common there are often not enough trained experts. Late or wrong diagnosis leads to untreated infections or unnecessary drug use.

### b) Application Domain

Healthcare. Specifically, computer-aided diagnosis in digital pathology and microscopy.

### c) Objective

The goal is an AI system that sorts single-cell images from thin blood smears into **Parasitized** or **Uninfected**. It is meant as a fast, consistent screening aid for laboratory technicians. The model will be a convolutional neural network trained on the public `malaria` dataset from TensorFlow Datasets (TFDS). It contains 27,558 labelled cell images, split evenly between the two classes, and comes from the U.S. National Library of Medicine (NIH/NLM) collection gathered at Chittagong Medical College Hospital, Bangladesh. Users upload a cell image through a web interface. They get a prediction with a confidence score and can give feedback on whether it was correct.

### d) Expected Outcome

1. A deployed, containerised application: a Streamlit front end and a FastAPI inference service.
2. A trained classifier aiming for at least 95% test accuracy, with high recall on the Parasitized class so that infections are rarely missed.
3. A reproducible, monitored pipeline that can retrain itself as user feedback comes in.

### e) AIOps Practices

The system will be built and run as a full machine-learning lifecycle, not a one-off model:

1. **Data ingestion and versioning (TFDS and DVC):** the dataset is pulled straight from TensorFlow Datasets. It is exported to a local data store, split into fixed train, validation and test sets, and versioned with DVC together with the preprocessed data and collected feedback. Every model can be traced back to the exact data that produced it, and the data preparation and training steps are written as reproducible DVC pipeline stages.
2. **Orchestration (Apache Airflow):** DAGs automate data ingestion, validation, preprocessing, training, evaluation, and retraining triggered by feedback.
3. **Experiment tracking and model registry (MLflow):** parameters, metrics and artifacts are logged for each run. Models are registered, and a new model is promoted to production only if it beats the current one.
4. **Containerisation (Docker Compose):** every service runs in its own container, so the environment is the same in development and deployment.
5. **Monitoring (Prometheus and Grafana):** dashboards track API latency, request volume, error rates, prediction-class distribution and confidence, to catch system problems and possible data drift.
6. **Feedback loop:** user corrections are stored, versioned with DVC, and fed back into scheduled retraining. This keeps the model improving after deployment.
