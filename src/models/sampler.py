# src/models/sampler.py

import random
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Iterator, Union


class PersonBatchSampler:
    """
    Person-aware batch sampler for medical examination data.
    
    Groups exams by person and shuffles persons (not individual exams) to maintain
    person-level sequence integrity while providing proper randomization for training.
    
    Supports fixed batch sizes with truncation - a person's exams may be split across
    batches if needed to maintain exact batch size requirements.
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize PersonBatchSampler.
        
        Args:
            manifest_path: Path to sorted manifest file (person_id, then ExamDate)
            batch_size: Target number of exams per batch
            shuffle: Whether to shuffle persons each epoch (True for train, False for val/test)
            drop_last: Whether to drop incomplete final batch
        """
        self.manifest_path = Path(manifest_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Build person-to-indices mapping once at initialization
        self.persons = self._build_person_index_map()
        
        # Cache total exams for __len__ calculation
        self.total_exams = sum(len(exam_indices) for _, exam_indices in self.persons)
    
    def _build_person_index_map(self) -> List[Tuple[str, List[int]]]:
        """
        Build person -> exam_indices mapping from sorted manifest.
        
        Returns:
            List of (person_id, exam_indices) tuples where indices are positional (0-based)
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        
        # Read manifest file
        df = pd.read_parquet(self.manifest_path)
        
        # CRITICAL: Reset index to ensure idx corresponds to positional row numbers
        # Manifest files may have custom indices like __index_level_0__ which don't
        # match the expected 0-based positional indices for Dataset.__getitem__()
        df = df.reset_index(drop=True)
        
        if 'person_id' not in df.columns:
            raise ValueError(f"Manifest file must contain 'person_id' column: {self.manifest_path}")
        
        person_groups = []
        current_person = None
        current_indices = []
        
        # Group consecutive exams by person_id
        # Now idx is guaranteed to be 0, 1, 2, ... matching Dataset expectations
        for idx, row in df.iterrows():
            person_id = row['person_id']
            
            if person_id != current_person:
                # Save previous person's exams
                if current_indices:
                    person_groups.append((current_person, current_indices))
                
                # Start new person
                current_person = person_id
                current_indices = [idx]
            else:
                # Add exam to current person
                current_indices.append(idx)
        
        # Add final person
        if current_indices:
            person_groups.append((current_person, current_indices))
        
        return person_groups
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches of exam indices.
        
        Yields:
            List of exam indices for each batch
        """
        # Shuffle persons if requested
        persons = self.persons.copy()
        if self.shuffle:
            random.shuffle(persons)
        
        # Track progress through persons and within current person
        person_idx = 0
        person_exam_offset = 0
        
        # Generate batches until all persons are processed
        while person_idx < len(persons):
            current_batch = []
            
            # Fill current batch to target size
            while len(current_batch) < self.batch_size and person_idx < len(persons):
                person_id, exam_indices = persons[person_idx]
                remaining_exams = exam_indices[person_exam_offset:]
                
                slots_available = self.batch_size - len(current_batch)
                
                if len(remaining_exams) <= slots_available:
                    # Use all remaining exams from this person
                    current_batch.extend(remaining_exams)
                    person_idx += 1
                    person_exam_offset = 0
                else:
                    # Truncate person's exams to fill batch exactly
                    current_batch.extend(remaining_exams[:slots_available])
                    person_exam_offset += slots_available
            
            # Yield batch if valid
            if len(current_batch) == self.batch_size:
                yield current_batch
            elif len(current_batch) > 0 and not self.drop_last:
                yield current_batch
    
    def __len__(self) -> int:
        """
        Calculate number of batches per epoch.
        
        Returns:
            Number of batches based on drop_last setting
        """
        if self.drop_last:
            return self.total_exams // self.batch_size
        else:
            return (self.total_exams + self.batch_size - 1) // self.batch_size
    
    def get_stats(self) -> dict:
        """
        Get sampler statistics for logging.
        
        Returns:
            Dictionary with sampler statistics
        """
        return {
            'total_persons': len(self.persons),
            'total_exams': self.total_exams,
            'avg_exams_per_person': self.total_exams / len(self.persons) if self.persons else 0,
            'batch_size': self.batch_size,
            'batches_per_epoch': len(self),
            'shuffle': self.shuffle,
            'drop_last': self.drop_last
        }
    
    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"PersonBatchSampler(persons={len(self.persons)}, "
                f"total_exams={self.total_exams}, batch_size={self.batch_size}, "
                f"shuffle={self.shuffle}, drop_last={self.drop_last})")


class InfinitePersonBatchSampler(PersonBatchSampler):
    """
    Person-aware *infinite* batch sampler.

    This class wraps ``PersonBatchSampler`` and turns it into an endless
    generator of person-aligned batches.  Each time the underlying sampler
    is exhausted, we restart it (optionally reshuffling persons if
    ``shuffle=True``) so that training loops driven by ``global_step``
    never run out of data.

    Notes
    -----
    • Semantic guarantees (person grouping, optional shuffling, drop_last)
      are identical to the parent class.  
    • ``__len__`` returns an extremely large integer because some tooling
      (e.g., tqdm) expects the attribute to exist, but its value is not
      used to control training.
    • Training loops should use ``while global_step < max_steps`` rather
      than iterating over the sampler directly.

    Example
    -------
    >>> sampler = InfinitePersonBatchSampler(
    ...     manifest_path="data/splits/core/sorted/train_ssl.parquet",
    ...     batch_size=32,
    ...     shuffle=True,
    ... )
    >>> loader = DataLoader(dataset, batch_sampler=sampler, num_workers=8)
    >>> for global_step, batch in enumerate(loader):
    ...     train_step(batch)
    ...     if global_step >= MAX_STEPS:
    ...         break
    """

    # Inherit __init__ from PersonBatchSampler; no changes needed.

    def __iter__(self) -> Iterator[List[int]]:
        """Yield person-aligned batches *forever*."""
        self.infinite = True
        while True:
            # ``super().__iter__()`` yields exactly one pass over the manifest
            for batch in super().__iter__():
                yield batch

    def __len__(self) -> int:  # type: ignore[override]
        """Return a very large number to denote 'practically infinite'."""
        return 2**63 - 1

    def get_stats(self) -> dict:
        """Return sampler statistics with human-friendly infinity."""
        stats = super().get_stats()
        stats['batches_per_epoch'] = '∞'
        stats['infinite'] = True
        return stats

    def __repr__(self) -> str:
        """String representation with key parameters."""
        return (f"InfinitePersonBatchSampler(persons={len(self.persons)}, "
                f"total_exams={self.total_exams}, batch_size={self.batch_size}, "
                f"shuffle={self.shuffle}, drop_last={self.drop_last})")