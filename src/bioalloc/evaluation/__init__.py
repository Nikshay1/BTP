"""The evaluation harness: task generators, reference answers and graders.

Nothing in the coordinator, the executors or the runtime checks may import this package; a test
enforces it. Reference answers are used only to grade final answers and, on the training split,
to produce training labels.
"""
