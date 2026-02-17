"""
Data Loader Module for Student Selection Prediction

Handles loading, preprocessing, and creating sequences from the Excel dataset.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
import pickle


class StudentSelectionDataset(Dataset):
    """PyTorch Dataset for student selection sequences."""
    
    def __init__(self, binary_vectors: np.ndarray, sequence_length: int = 14):
        """
        Initialize dataset.
        
        Args:
            binary_vectors: Binary matrix (n_samples, n_students)
            sequence_length: Number of past days to use as input
        """
        self.data = binary_vectors
        self.sequence_length = sequence_length
        self.n_students = binary_vectors.shape[1]
        
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input: past sequence_length days
        X = self.data[idx:idx + self.sequence_length]
        # Target: next day's selection
        y = self.data[idx + self.sequence_length]
        
        return torch.FloatTensor(X), torch.FloatTensor(y)


class DataManager:
    """Manages data loading, preprocessing, and dataset creation."""
    
    def __init__(self, excel_path: str = 'Database.xlsx', total_students: int = 55):
        """
        Initialize data manager.
        
        Args:
            excel_path: Path to Excel file
            total_students: Total number of students
        """
        self.excel_path = excel_path
        self.total_students = total_students
        self.data = None
        self.binary_vectors = None
        self.days = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from Excel file."""
        print(f"Loading data from {self.excel_path}...")
        self.data = pd.read_excel(self.excel_path)
        print(f"Loaded {len(self.data)} rows")
        return self.data
    
    def convert_to_binary(self) -> np.ndarray:
        """Convert student ID columns to binary vectors."""
        print("Converting to binary vectors...")
        
        student_columns = [col for col in self.data.columns if col != 'Days']
        n_rows = len(self.data)
        binary_matrix = np.zeros((n_rows, self.total_students), dtype=np.float32)
        
        for idx, row in self.data.iterrows():
            for col in student_columns:
                student_id = row[col]
                if pd.notna(student_id):
                    student_id = int(student_id)
                    if 1 <= student_id <= self.total_students:
                        binary_matrix[idx, student_id - 1] = 1
        
        self.binary_vectors = binary_matrix
        self.days = self.data['Days'].values
        
        print(f"Created binary vectors: {binary_matrix.shape}")
        return binary_matrix
    
    def sort_by_days(self):
        """Sort data chronologically by Days column."""
        print("Sorting by days...")
        sorted_indices = self.data['Days'].argsort()
        self.data = self.data.iloc[sorted_indices].reset_index(drop=True)
        self.binary_vectors = self.binary_vectors[sorted_indices]
        self.days = self.days[sorted_indices]
        print(f"Sorted. Days range: {self.days.min()} to {self.days.max()}")
    
    def create_datasets(self, sequence_length: int = 14, 
                       train_ratio: float = 0.9) -> Tuple[StudentSelectionDataset, StudentSelectionDataset]:
        """
        Create train and validation datasets.
        
        Args:
            sequence_length: Number of past days to use as input
            train_ratio: Ratio of data to use for training
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if self.binary_vectors is None:
            raise ValueError("Must load and convert data first")
        
        split_idx = int(len(self.binary_vectors) * train_ratio)
        
        train_data = self.binary_vectors[:split_idx]
        val_data = self.binary_vectors[split_idx:]
        
        train_dataset = StudentSelectionDataset(train_data, sequence_length)
        val_dataset = StudentSelectionDataset(val_data, sequence_length)
        
        print(f"Created datasets:")
        print(f"  Train: {len(train_dataset)} sequences")
        print(f"  Validation: {len(val_dataset)} sequences")
        
        return train_dataset, val_dataset
    
    def get_recent_sequence(self, n_days: int = 14) -> np.ndarray:
        """
        Get the most recent n days as a sequence.
        
        Args:
            n_days: Number of recent days to retrieve
            
        Returns:
            Binary matrix of shape (n_days, n_students)
        """
        if self.binary_vectors is None:
            raise ValueError("Must load data first")
        
        return self.binary_vectors[-n_days:]
    
    def append_new_selection(self, selected_students: List[int], day_number: Optional[int] = None) -> None:
        """
        Append a new selection to the dataset (for online learning).
        
        Args:
            selected_students: List of selected student IDs (1-indexed)
            day_number: Specific day number (if None, auto-increment)
        """
        new_binary = np.zeros(self.total_students, dtype=np.float32)
        for student_id in selected_students:
            if 1 <= student_id <= self.total_students:
                new_binary[student_id - 1] = 1
        
        self.binary_vectors = np.vstack([self.binary_vectors, new_binary])
        
        # Update days
        if day_number is None:
            next_day = self.days[-1] + 1 if len(self.days) > 0 else 1
        else:
            next_day = day_number
        self.days = np.append(self.days, next_day)
        
        print(f"Appended new selection. Total samples: {len(self.binary_vectors)}")
    
    def save_to_excel(self, output_path: Optional[str] = None) -> None:
        """
        Save current data back to Excel file.
        
        Args:
            output_path: Path to save Excel file (if None, use original path)
        """
        if output_path is None:
            output_path = self.excel_path
        
        # Convert binary vectors back to student IDs
        rows = []
        for day_idx, binary_vector in enumerate(self.binary_vectors):
            selected_indices = np.where(binary_vector == 1)[0]
            student_ids = [int(idx + 1) for idx in selected_indices]
            
            # Pad with None if less than 6 students (shouldn't happen)
            while len(student_ids) < 6:
                student_ids.append(None)
            
            row = {
                'Days': int(self.days[day_idx]),
                'Student ID 1': student_ids[0] if len(student_ids) > 0 else None,
                'Student ID 2': student_ids[1] if len(student_ids) > 1 else None,
                'Student ID 3': student_ids[2] if len(student_ids) > 2 else None,
                'Student ID 4': student_ids[3] if len(student_ids) > 3 else None,
                'Student ID 5': student_ids[4] if len(student_ids) > 4 else None,
                'Student ID 6': student_ids[5] if len(student_ids) > 5 else None,
            }
            rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(rows)
        df.to_excel(output_path, index=False)
        print(f"💾 Database saved to {output_path}")
    
    def save(self, filepath: str = 'data_manager.pkl'):
        """Save data manager state."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'binary_vectors': self.binary_vectors,
                'days': self.days,
                'total_students': self.total_students
            }, f)
        print(f"Saved data manager to {filepath}")
    
    def load(self, filepath: str = 'data_manager.pkl'):
        """Load data manager state."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.binary_vectors = data['binary_vectors']
        self.days = data['days']
        self.total_students = data['total_students']
        print(f"Loaded data manager from {filepath}")


def prepare_data_pipeline(excel_path: str = 'Database.xlsx', 
                         sequence_length: int = 14,
                         train_ratio: float = 0.9) -> Tuple[DataManager, StudentSelectionDataset, StudentSelectionDataset]:
    """
    Complete data preparation pipeline.
    
    Args:
        excel_path: Path to Excel file
        sequence_length: Sequence length for model
        train_ratio: Train/val split ratio
        
    Returns:
        Tuple of (data_manager, train_dataset, val_dataset)
    """
    data_manager = DataManager(excel_path)
    data_manager.load_data()
    data_manager.convert_to_binary()
    data_manager.sort_by_days()
    
    train_dataset, val_dataset = data_manager.create_datasets(
        sequence_length=sequence_length,
        train_ratio=train_ratio
    )
    
    return data_manager, train_dataset, val_dataset


if __name__ == "__main__":
    # Test the data pipeline
    print("="*60)
    print("TESTING DATA PIPELINE")
    print("="*60)
    
    data_manager, train_dataset, val_dataset = prepare_data_pipeline(
        excel_path='Database.xlsx',
        sequence_length=14,
        train_ratio=0.9
    )
    
    # Test dataset
    print(f"\nTesting dataset...")
    X, y = train_dataset[0]
    print(f"Input shape: {X.shape}")  # (sequence_length, n_students)
    print(f"Target shape: {y.shape}")  # (n_students,)
    
    # Test DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    batch_X, batch_y = next(iter(train_loader))
    print(f"\nBatch input shape: {batch_X.shape}")
    print(f"Batch target shape: {batch_y.shape}")
    
    print("\n✓ Data pipeline test complete")
