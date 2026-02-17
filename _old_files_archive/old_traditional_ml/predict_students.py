"""
Student Selection Prediction - Output Top Candidates

This script loads a trained model and predicts the next 6 students to be selected,
along with showing the top 20 candidates with probability scores.
"""

import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

def load_model_and_scaler(model_name='xgboost'):
    """Load trained model and scaler."""
    print(f"Loading {model_name} model...")
    
    # Load scaler
    with open('trained_model_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load model
    if model_name == 'neural_network':
        # Load PyTorch model
        class MultiLabelNN(nn.Module):
            def __init__(self, input_size=4019, hidden_layers=[256, 128, 64], output_size=55):
                super(MultiLabelNN, self).__init__()
                layers_list = []
                
                prev_size = input_size
                for units in hidden_layers:
                    layers_list.append(nn.Linear(prev_size, units))
                    layers_list.append(nn.ReLU())
                    layers_list.append(nn.BatchNorm1d(units))
                    layers_list.append(nn.Dropout(0.3))
                    prev_size = units
                
                layers_list.append(nn.Linear(prev_size, output_size))
                layers_list.append(nn.Sigmoid())
                
                self.network = nn.Sequential(*layers_list)
            
            def forward(self, x):
                return self.network(x)
        
        model = MultiLabelNN()
        model.load_state_dict(torch.load('trained_model_neural_network.pth'))
        model.eval()
    else:
        model_file = f'trained_model_{model_name}.pkl'
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    
    return model, scaler


def predict_probabilities(model, scaler, features, model_name='xgboost'):
    """Predict probabilities for all 55 students."""
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predict
    if model_name == 'neural_network':
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled)
            probabilities = model(features_tensor).numpy()[0]
    else:
        probabilities = model.predict_proba(features_scaled)[0]
    
    return probabilities


def display_predictions(probabilities, top_k=20, select_k=6):
    """Display top candidates and selected students."""
    
    # Get top-k indices sorted by probability
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_student_ids = top_indices + 1  # Convert to 1-indexed
    top_probs = probabilities[top_indices]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Rank': range(1, top_k + 1),
        'Student ID': top_student_ids,
        'Probability': top_probs,
        'Selected': ['✓' if i < select_k else '' for i in range(top_k)]
    })
    
    return results_df, top_student_ids[:select_k], top_probs[:select_k]


def main():
    """Main prediction function."""
    print("="*70)
    print("STUDENT SELECTION PREDICTION")
    print("="*70)
    
    # Load test features (use last sample as example)
    print("\nLoading test features...")
    X_test = np.load('X_test_features.npy')
    Y_test = np.load('Y_test_labels.npy')
    
    # Use the last test sample for prediction
    current_features = X_test[-1]
    actual_selection = Y_test[-1]
    actual_students = [i+1 for i, val in enumerate(actual_selection) if val == 1]
    
    print(f"Number of test samples: {len(X_test)}")
    print(f"Using last test sample for prediction")
    print(f"\nActual selected students: {actual_students}")
    
    # Try all models
    models_to_try = ['xgboost', 'random_forest', 'logistic_regression', 'neural_network']
    
    for model_name in models_to_try:
        try:
            print("\n" + "="*70)
            print(f"MODEL: {model_name.upper().replace('_', ' ')}")
            print("="*70)
            
            # Load model
            model, scaler = load_model_and_scaler(model_name)
            
            # Predict probabilities
            probabilities = predict_probabilities(model, scaler, current_features, model_name)
            
            # Display results
            results_df, selected_students, selected_probs = display_predictions(
                probabilities, top_k=20, select_k=6
            )
            
            # Print top 6 selected students (highlighted)
            print("\n🎯 SELECTED STUDENTS (Top 6):")
            print("-" * 70)
            for i in range(6):
                student_id = results_df.iloc[i]['Student ID']
                prob = results_df.iloc[i]['Probability']
                match = "✓✓✓" if student_id in actual_students else "   "
                print(f"  {match} #{i+1}: Student {student_id:2d} - Probability: {prob:.4f}")
            
            # Calculate overlap with actual
            overlap = len(set(selected_students) & set(actual_students))
            print(f"\n  Overlap with actual: {overlap} / 6 students")
            
            # Print top 20 candidates
            print(f"\n📊 TOP 20 CANDIDATES:")
            print("-" * 70)
            print(f"{'Rank':<6} {'Student ID':<12} {'Probability':<14} {'Selected':<10} {'Match':<6}")
            print("-" * 70)
            
            for _, row in results_df.iterrows():
                student_id = row['Student ID']
                match = "✓" if student_id in actual_students else ""
                selected_mark = row['Selected']
                print(f"{row['Rank']:<6} {student_id:<12} {row['Probability']:<14.6f} {selected_mark:<10} {match:<6}")
            
            # Save results for this model
            output_file = f'predictions_{model_name}.csv'
            results_df.to_csv(output_file, index=False)
            print(f"\n💾 Saved: {output_file}")
            
        except FileNotFoundError as e:
            print(f"Model {model_name} not found, skipping...")
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    # Summary comparison
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print("\nAll models predict probabilities for each of the 55 students.")
    print("The top 6 students with highest probabilities are selected.")
    print("\n✓ = Student was actually selected")
    print("Tree-based models (XGBoost, Random Forest) tend to be most reliable.")


def predict_for_new_data():
    """Example function for predicting with new feature data."""
    print("\n" + "="*70)
    print("EXAMPLE: Predicting for custom features")
    print("="*70)
    
    # Load the best model
    model, scaler = load_model_and_scaler('xgboost')
    
    # Load latest features
    X_test = np.load('X_test_features.npy')
    
    # Predict for each test sample
    print("\nPredicting for all test samples...")
    all_predictions = []
    
    for i in range(min(5, len(X_test))):  # Show first 5 samples
        probabilities = predict_probabilities(model, scaler, X_test[i], 'xgboost')
        top_6_indices = np.argsort(probabilities)[-6:][::-1]
        top_6_students = top_6_indices + 1
        top_6_probs = probabilities[top_6_indices]
        
        all_predictions.append({
            'Sample': i+1,
            'Predicted_Students': list(top_6_students),
            'Probabilities': list(top_6_probs)
        })
    
    predictions_df = pd.DataFrame(all_predictions)
    print("\nFirst 5 predictions:")
    for _, row in predictions_df.iterrows():
        students = ', '.join([str(s) for s in row['Predicted_Students']])
        print(f"  Sample {row['Sample']}: Students [{students}]")
    
    return predictions_df


if __name__ == "__main__":
    main()
    predict_for_new_data()
