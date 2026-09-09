import unittest

from src.search import _lexical_overlap
from src.serving import LRUCache, assign_variant, cache_key


class ServingTests(unittest.TestCase):
    def test_lexical_overlap(self):
        self.assertEqual(_lexical_overlap("wireless mouse", "wireless gaming mouse"), 1.0)
        self.assertEqual(_lexical_overlap("wireless mouse", "mechanical keyboard"), 0.0)

    def test_lru_cache(self):
        cache = LRUCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        self.assertEqual(cache.get("a"), 1)
        cache.set("c", 3)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.size, 2)

    def test_cache_key_is_order_independent(self):
        self.assertEqual(cache_key({"a": 1, "b": 2}), cache_key({"b": 2, "a": 1}))

    def test_ab_assignment_is_stable(self):
        self.assertEqual(assign_variant("same-user"), assign_variant("same-user"))
        self.assertIn(assign_variant("another-user"), {"dense", "enhanced"})


if __name__ == "__main__":
    unittest.main()
