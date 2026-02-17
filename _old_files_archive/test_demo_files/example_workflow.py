"""
Example usage of the Prediction API

This script demonstrates the complete workflow:
1. Make prediction for day 1309
2. Submit correction later
3. Make prediction for day 1310
4. View status
5. Update model
"""

from prediction_api import PredictionAPI

def main():
    print("="*70)
    print("EXAMPLE: PREDICTION API WORKFLOW")
    print("="*70)
    
    # Initialize API
    print("\n🔧 Initializing API...")
    api = PredictionAPI()
    
    # Example 1: Make prediction for day 1311
    print("\n" + "="*70)
    print("STEP 1: Make Prediction (Before Class)")
    print("="*70)
    
    prediction = api.predict(day_number=1311, n_groups=5)
    
    print(f"\n📋 Prediction Summary:")
    print(f"   Day: {prediction['day']}")
    print(f"   Number of groups: {len(prediction['predicted_groups'])}")
    print(f"   Status: {prediction['status']}")
    
    # Example 2: Simulate class happening, then submit correction
    print("\n" + "="*70)
    print("STEP 2: Submit Correction (After Class)")
    print("="*70)
    print("\n(Simulating class has happened...)")
    print("Teacher selected: [3, 8, 13, 18, 23, 28]")
    
    correction = api.correct(
        day_number=1311,
        actual_students=[3, 8, 13, 18, 23, 28],
        auto_update=False
    )
    
    print(f"\n📊 Correction Summary:")
    print(f"   Day: {correction['day']}")
    print(f"   Best group: Group {correction['best_group_index'] + 1}")
    print(f"   Best overlap: {correction['best_overlap']}/6")
    print(f"   Best accuracy: {correction['best_accuracy']:.1%}")
    print(f"   Status: {correction['status']}")
    
    # Example 3: Make another prediction
    print("\n" + "="*70)
    print("STEP 3: Make Another Prediction")
    print("="*70)
    
    prediction2 = api.predict(day_number=1312, n_groups=5)
    
    print(f"\n📋 Prediction Summary:")
    print(f"   Day: {prediction2['day']}")
    print(f"   First group: {prediction2['predicted_groups'][0]}")
    print(f"   Status: {prediction2['status']}")
    
    # Example 4: Check status
    print("\n" + "="*70)
    print("STEP 4: Check Overall Status")
    print("="*70)
    
    api.status()
    
    # Example 5: Show how to update model
    print("\n" + "="*70)
    print("STEP 5: Update Model (Optional)")
    print("="*70)
    
    print("\n💡 To update model with all corrections:")
    print("   api.update_model()")
    print("\n💡 Or from command line:")
    print("   python prediction_api.py update")
    
    # Summary
    print("\n" + "="*70)
    print("✅ WORKFLOW COMPLETE")
    print("="*70)
    
    print("\n📝 What we did:")
    print("   1. ✅ Made prediction for day 1311 (5 groups)")
    print("   2. ✅ Submitted correction for day 1311")
    print("   3. ✅ Made prediction for day 1312 (5 groups)")
    print("   4. ✅ Checked overall status")
    print("   5. 💡 Model can be updated when ready")
    
    print("\n🎯 Next steps:")
    print("   • Wait for class on day 1312")
    print("   • Submit correction for day 1312")
    print("   • Update model to learn from corrections")
    
    print("\n📁 Files created:")
    print("   • predictions/prediction_day_1311.json")
    print("   • predictions/correction_day_1311.json")
    print("   • predictions/prediction_day_1312.json")
    print("   • predictions/predictions_summary.csv")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
