import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os

def parallel_process(func, items, num_workers=None):
    """
    Process items in parallel without requiring the user to handle multiprocessing details.
    
    Args:
        func: The function to apply to each item
        items: List of items to process
        num_workers: Number of worker processes (defaults to CPU count)
        
    Returns:
        List of results in the same order as input items
    """
    if num_workers is None:
        num_workers = os.cpu_count()
    
    # Use ProcessPoolExecutor which handles the process spawning internally
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(func, items))
    
    return results

# Example usage - this can be imported and used directly without if __name__ == "__main__"
def example():
    # Define a simple operation
    def square(x):
        return x * x
    
    # Sample data
    data = list(range(10))
    
    # Process in parallel
    results = parallel_process(square, data)
    print(f"Results: {results}")
    
    return results

# This function can be safely imported and used anywhere without if __name__ == "__main__"

example()