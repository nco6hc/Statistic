"""
Student Selection Data Processing Script

This script processes an Excel dataset containing student selection records,
converts them to binary vectors, and splits them into train/test sets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


class StudentDataProcessor:
    """Processes student selection data from Excel into binary vectors."""
    
    def __init__(self, file_path: str, total_students: int = 55):
        """
        Initialize the processor.
        
        Args:
            file_path: Path to the Excel file
            total_students: Total number of students in the class
        """
        self.file_path = file_path
        self.total_students = total_students
        self.data = None
        self.binary_vectors = None
        self.days = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from Excel file."""
        print(f"Loading data from {self.file_path}...")
        self.data = pd.read_excel(self.file_path)
        print(f"Loaded {len(self.data)} rows")
        return self.data
    
    def convert_to_binary_vectors(self) -> np.ndarray:
        """
        Convert student ID columns to binary vectors.
        
        Returns:
            Binary matrix of shape (n_samples, total_students)
        """
        print("Converting to binary vectors...")
        
        # Get student ID columns (all columns except 'Days')
        student_columns = [col for col in self.data.columns if col != 'Days']
        
        # Initialize binary matrix
        n_rows = len(self.data)
        binary_matrix = np.zeros((n_rows, self.total_students), dtype=int)
        
        # Fill binary matrix
        for idx, row in self.data.iterrows():
            for col in student_columns:
                student_id = row[col]
                # Handle NaN values (empty cells)
                if pd.notna(student_id):
                    student_id = int(student_id)
                    # Validate student ID range
                    if 1 <= student_id <= self.total_students:
                        binary_matrix[idx, student_id - 1] = 1
                    else:
                        print(f"Warning: Student ID {student_id} out of range at row {idx}")
        
        self.binary_vectors = binary_matrix
        print(f"Created binary vectors with shape: {binary_matrix.shape}")
        return binary_matrix
    
    def sort_by_days(self) -> None:
        """Sort data by Days column."""
        print("Sorting data by Days...")
        
        # Sort indices by Days
        sorted_indices = self.data['Days'].argsort()
        
        # Apply sorting
        self.data = self.data.iloc[sorted_indices].reset_index(drop=True)
        self.binary_vectors = self.binary_vectors[sorted_indices]
        self.days = self.data['Days'].values
        
        print(f"Data sorted. Days range: {self.days.min()} to {self.days.max()}")
    
    def split_train_test(self, test_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            test_size: Proportion of data to use for testing (default: 0.1 for last 10%)
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            where y represents the Days values
        """
        print(f"Splitting data with {test_size*100}% as test set...")
        
        # Calculate split point (last 10%)
        split_idx = int(len(self.binary_vectors) * (1 - test_size))
        
        # Split the data (chronological split, not random)
        X_train = self.binary_vectors[:split_idx]
        X_test = self.binary_vectors[split_idx:]
        y_train = self.days[:split_idx]
        y_test = self.days[split_idx:]
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def get_summary_statistics(self) -> dict:
        """Get summary statistics of the binary vectors."""
        if self.binary_vectors is None:
            raise ValueError("Binary vectors not created yet. Run convert_to_binary_vectors() first.")
        
        stats = {
            'total_samples': len(self.binary_vectors),
            'students_per_selection_mean': self.binary_vectors.sum(axis=1).mean(),
            'students_per_selection_std': self.binary_vectors.sum(axis=1).std(),
            'student_selection_frequency': self.binary_vectors.sum(axis=0),
        }
        return stats


def main():
    """Main execution function."""
    # Configuration
    FILE_PATH = 'Database.xlsx'
    TOTAL_STUDENTS = 55
    TEST_SIZE = 0.1
    
    # Initialize processor
    processor = StudentDataProcessor(FILE_PATH, TOTAL_STUDENTS)
    
    # Step 1: Load data
    data = processor.load_data()
    print(f"\nColumns: {list(data.columns)}")
    print(f"First few rows:\n{data.head()}\n")
    
    # Step 2: Convert to binary vectors
    binary_vectors = processor.convert_to_binary_vectors()
    
    # Step 3: Sort by Days
    processor.sort_by_days()
    
    # Step 4: Split into train/test
    X_train, X_test, y_train, y_test = processor.split_train_test(test_size=TEST_SIZE)
    
    # Display summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    stats = processor.get_summary_statistics()
    print(f"Total samples: {stats['total_samples']}")
    print(f"Average students per selection: {stats['students_per_selection_mean']:.2f} ± {stats['students_per_selection_std']:.2f}")
    print(f"\nMost frequently selected students:")
    top_students = np.argsort(stats['student_selection_frequency'])[::-1][:5]
    for i, student_idx in enumerate(top_students, 1):
        print(f"  {i}. Student {student_idx+1}: {stats['student_selection_frequency'][student_idx]} times")
    
    print("\n" + "="*50)
    print("DATA SPLIT COMPLETE")
    print("="*50)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Days range (train): {y_train.min()} to {y_train.max()}")
    print(f"Days range (test): {y_test.min()} to {y_test.max()}")
    
    # Optional: Save processed data
    save_processed = input("\nSave processed data to files? (y/n): ")
    if save_processed.lower() == 'y':
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
        print("Saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
    
    return processor, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    processor, X_train, X_test, y_train, y_test = main()
