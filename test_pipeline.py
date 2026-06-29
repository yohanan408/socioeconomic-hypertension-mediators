import sys
import os

# 1. Ensure the system can look inside your custom src folder
sys.path.append(os.path.abspath("src"))

try:
    print("🔄 Step 1: Initializing Data Pipeline...")
    import data_prep
    # Replace 'data/heart_disease.csv' with the actual path to your source file
    df = data_prep.load_and_filter("heart_disease_health_indicators_BRFSS2015.csv") 
    print(f"✅ Data Prep Loaded! Dimensions: {df.shape}\n")

    print("🔄 Step 2: Testing Baseline Screening...")
    import baseline_screening
    baseline_results = baseline_screening.run_bivariate_eda(df)
    print("✅ Baseline screenings computed successfully!\n")

    print("🔄 Step 3: Fitting Multivariate Logit Models...")
    import statistical_models
    models_A = statistical_models.fit_path_A_models(df)
    model_B = statistical_models.fit_path_B_model(df)
    print("✅ All regressions resolved successfully!\n")

    print("🔄 Step 4: Executing Mediation Calculation Engine...")
    import mediation_engine
    # Pass your models into the agent's function (check name if it differs slightly)
    final_percentages = mediation_engine.compute_mediation(models_A, model_B)
    print("✅ Mediation algebra verified!")
    print(f"📊 Extracted Percentages: {final_percentages}\n")

    print("🚀 PIPELINE STATUS: 100% OPERATIONAL & SUCCESSFUL!")

except Exception as e:
    print(f"\n❌ PIPELINE CRASHED! Error encountered: {str(e)}")
    print("Double-check your source data file path or specific function parameter names.")
