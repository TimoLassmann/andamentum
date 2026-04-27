# Stack Overflow: How do I parallelize a for loop in Python?

**asked by user_42**

I have a Python for loop that processes a large list. Each iteration is independent. I'd like to run them in parallel to speed things up. What's the simplest way?

**answered by senior_dev**

The easiest option is `concurrent.futures.ProcessPoolExecutor`. It hides most of the multiprocessing boilerplate. Here's the pattern:

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor() as ex:
    results = list(ex.map(process_item, items))
```

If your iterations are I/O-bound rather than CPU-bound, swap `ProcessPoolExecutor` for `ThreadPoolExecutor`.

**asked by user_42**

Thanks. Will this work if `process_item` returns a custom class?

**answered by senior_dev**

Yes, as long as the class is picklable. Anything defined at module level with no unpicklable attributes (no open file handles, no lambdas) will work fine. If you hit pickling errors, the simplest fix is usually to return a plain dict from `process_item` and reconstruct the class in the parent process.
