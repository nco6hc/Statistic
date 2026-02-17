"""
Cleanup Script - Remove Unnecessary Files

This script identifies and optionally removes old/redundant files
that have been superseded by the new deep learning pipeline.
"""

import os
import shutil
from pathlib import Path

def get_files_to_cleanup():
    """Identify files that can be safely removed."""
    
    cleanup_categories = {
        'old_traditional_ml': [
            'predict_students.py',  # Old prediction script (traditional ML)
            'train_models.py',  # Old training with XGBoost/RF/LR
            'markov_chain_model.py',  # Old Markov Chain approach
            'process_student_data.py',  # Old data processing
            'feature_engineering.py',  # Old feature engineering (4019 features)
            'model_summary.py',  # Old results visualization
        ],
        
        'old_model_files': [
            'trained_model_xgboost.pkl',
            'trained_model_random_forest.pkl',
            'trained_model_logistic_regression.pkl',
            'trained_model_neural_network.pth',
            'trained_model_scaler.pkl',
            'markov_model.pkl',
            'model_comparison.csv',
        ],
        
        'old_data_files': [
            'X_train.npy',
            'X_test.npy',
            'y_train.npy',
            'y_test.npy',
            'X_train_features.npy',
            'X_test_features.npy',
            'Y_train_labels.npy',
            'Y_test_labels.npy',
            'feature_names.txt',
        ],
        
        'test_demo_files': [
            'test_multi_groups.py',
            'demo_multigroup.py',
            'show_multigroup_summary.py',
            'example_workflow.py',
            'test_randomness.py',  # Analysis file, not core
            'randomness_test_results.csv',
        ],
        
        'demo_checkpoints': [
            'demo_checkpoints/',  # Demo checkpoint directory
        ],
        
        'old_prediction_outputs': [
            'predictions_xgboost.csv',
            'predictions_random_forest.csv',
            'predictions_logistic_regression.csv',
            'predictions_neural_network.csv',
        ],
    }
    
    return cleanup_categories


def analyze_disk_usage(files_dict):
    """Calculate disk space that can be freed."""
    total_size = 0
    file_count = 0
    
    for category, files in files_dict.items():
        for file_path in files:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    for root, dirs, files in os.walk(file_path):
                        for f in files:
                            fp = os.path.join(root, f)
                            if os.path.exists(fp):
                                total_size += os.path.getsize(fp)
                                file_count += 1
                else:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
    
    return total_size, file_count


def create_archive(files_dict, archive_dir='_old_files_archive'):
    """Move files to archive instead of deleting."""
    archive_path = Path(archive_dir)
    archive_path.mkdir(exist_ok=True)
    
    moved_files = []
    
    for category, files in files_dict.items():
        category_dir = archive_path / category
        category_dir.mkdir(exist_ok=True)
        
        for file_path in files:
            if os.path.exists(file_path):
                try:
                    dest = category_dir / os.path.basename(file_path)
                    if os.path.isdir(file_path):
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(file_path, dest)
                        shutil.rmtree(file_path)
                    else:
                        shutil.move(file_path, dest)
                    moved_files.append(file_path)
                    print(f"✓ Moved: {file_path} → {category}/")
                except Exception as e:
                    print(f"✗ Error moving {file_path}: {e}")
    
    return moved_files


def delete_files(files_dict):
    """Permanently delete files."""
    deleted_files = []
    
    for category, files in files_dict.items():
        for file_path in files:
            if os.path.exists(file_path):
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                    deleted_files.append(file_path)
                    print(f"✓ Deleted: {file_path}")
                except Exception as e:
                    print(f"✗ Error deleting {file_path}: {e}")
    
    return deleted_files


def main():
    """Main cleanup function."""
    print("="*70)
    print("CLEANUP ANALYSIS")
    print("="*70)
    
    cleanup_files = get_files_to_cleanup()
    
    # Analyze what exists
    print("\n📊 Files to Clean Up:\n")
    
    total_exists = 0
    for category, files in cleanup_files.items():
        exists_in_category = [f for f in files if os.path.exists(f)]
        if exists_in_category:
            print(f"\n{category.upper().replace('_', ' ')}:")
            for file_path in exists_in_category:
                if os.path.isdir(file_path):
                    print(f"  📁 {file_path}")
                else:
                    size = os.path.getsize(file_path) / 1024  # KB
                    print(f"  📄 {file_path} ({size:.1f} KB)")
                total_exists += 1
    
    if total_exists == 0:
        print("\n✅ No unnecessary files found - workspace is clean!")
        return
    
    # Calculate disk space
    total_size, file_count = analyze_disk_usage(cleanup_files)
    size_mb = total_size / (1024 * 1024)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTotal files to clean: {total_exists}")
    print(f"Disk space to free: {size_mb:.2f} MB")
    
    print("\n" + "="*70)
    print("OPTIONS")
    print("="*70)
    print("\n1. ARCHIVE - Move files to '_old_files_archive/' (safe, can restore)")
    print("2. DELETE - Permanently delete files")
    print("3. CANCEL - Keep everything")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\n📦 Archiving files...")
        moved = create_archive(cleanup_files)
        print(f"\n✅ Archived {len(moved)} items to '_old_files_archive/'")
        print("   (You can delete this folder later if not needed)")
    
    elif choice == '2':
        confirm = input("\n⚠️  Permanently delete? Type 'yes' to confirm: ").strip().lower()
        if confirm == 'yes':
            print("\n🗑️  Deleting files...")
            deleted = delete_files(cleanup_files)
            print(f"\n✅ Deleted {len(deleted)} items")
        else:
            print("\n❌ Cancelled")
    
    else:
        print("\n❌ Cancelled - no changes made")
    
    # Show what to keep
    print("\n" + "="*70)
    print("KEEPING (Essential Files)")
    print("="*70)
    print("\n✅ Core Application:")
    print("  • prediction_api.py - Main API")
    print("  • dl_pipeline/ - Deep learning pipeline")
    print("  • Database.xlsx - Training data")
    print("  • visualizations.py - Visualization tool")
    
    print("\n✅ Documentation:")
    print("  • README files (*.md)")
    print("  • PREDICTION_API_GUIDE.md")
    print("  • MULTIGROUP_UPDATE.md")
    
    print("\n✅ Active Data:")
    print("  • predictions/ - Active predictions and corrections")
    print("  • dl_pipeline/dl_checkpoints/ - Trained models")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
