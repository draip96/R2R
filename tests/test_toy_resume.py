"""Resume bookkeeping checks for extended ToyMemory runs."""

import tempfile
import unittest

from recall2imagine import embodied


class ToyResumeTest(unittest.TestCase):

  def test_old_checkpoint_preserves_summary_seeded_update_count(self):
    with tempfile.TemporaryDirectory() as directory:
      path = embodied.Path(directory) / 'checkpoint.ckpt'
      old = embodied.Checkpoint(path, parallel=False, log=False)
      old.step = embodied.Counter(25000)
      old.save()

      resumed = embodied.Checkpoint(path, parallel=False, log=False)
      step = embodied.Counter()
      updates = embodied.Counter(5226)
      resumed.step = step
      resumed.updates = updates
      resumed.load()
      self.assertEqual(int(step), 25000)
      self.assertEqual(int(updates), 5226)

      updates.increment(7)
      resumed.save()
      restored = embodied.Checkpoint(path, parallel=False, log=False)
      restored_step = embodied.Counter()
      restored_updates = embodied.Counter()
      restored.step = restored_step
      restored.updates = restored_updates
      restored.load()
      self.assertEqual(int(restored_step), 25000)
      self.assertEqual(int(restored_updates), 5233)


if __name__ == '__main__':
  unittest.main()
