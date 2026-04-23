.PHONY: all data scrape merge validate clean features train train-smote-tomek train-smote train-borderline train-single train-single-smote-tomek train-single-smote train-single-borderline compare-models mlflow-ui test lint eda install-ml

#  Default 
all: data validate clean features train test

#  Data Acquisition & Merge 
data: scrape merge

scrape:
	python scraper/scraper.py

merge:
	python scraper/merge.py

#  Validation 
validate:
	python validation/validation.py --data data/merged/merged_airbnb_data.csv

#  Cleaning 
clean:
	python cleaning/cleaning.py

#  Feature Engineering 
features:
	python feature_engineering/pipeline.py

#  Model Training 
# Train all models (full hyperparameter grids)
train:
	python modelling/train.py

train-smote-tomek:
	python modelling/train_enhanced.py --balance smote_tomek

train-smote:
	python modelling/train_enhanced.py --balance mild_smote

train-borderline:
	python modelling/train_enhanced.py --balance borderline

# Train all models (single parameter sets - fastest)
train-single:
	python modelling/train_single.py

train-single-smote-tomek:
	python modelling/train_single.py --balance smote_tomek

train-single-smote:
	python modelling/train_single.py --balance mild_smote

train-single-borderline:
	python modelling/train_single.py --balance borderline

compare-models:
	python modelling/compare_models.py

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000 --default-artifact-root ./mlruns


#  EDA Dashboard 
eda:
	streamlit run eda/dashboard.py

#  Testing 
test:
	pytest tests/ -v --cov=. --cov-report=term-missing

#  Linting 
lint:
	ruff check .
	ruff format --check .
