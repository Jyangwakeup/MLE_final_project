import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments import consolidate_worktrees as c


class ConsolidationSafetyTests(unittest.TestCase):
    def test_conflicting_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            source,target=Path(d)/'source',Path(d)/'target'
            source.write_bytes(b'new');target.write_bytes(b'old')
            with self.assertRaises(ValueError):c.checked_copy(source,target)
            self.assertEqual(target.read_bytes(),b'old')

    def test_same_relative_name_is_preserved_in_each_origin(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name,data in [('one',b'one'),('two',b'two')]:
                source=root/name/'logs/game.log';source.parent.mkdir(parents=True);source.write_bytes(data)
                c.checked_copy(source,root/'archive'/name/'logs/game.log')
            self.assertEqual((root/'archive/one/logs/game.log').read_bytes(),b'one')
            self.assertEqual((root/'archive/two/logs/game.log').read_bytes(),b'two')

    def test_repeat_copy_preserves_identity_and_content(self):
        with tempfile.TemporaryDirectory() as d:
            source,target=Path(d)/'source',Path(d)/'target';source.write_bytes(b'stable')
            first=c.checked_copy(source,target);before=c.identity(target)
            self.assertEqual(c.checked_copy(source,target),first)
            self.assertEqual(c.identity(target),before)

    def test_changed_source_prevents_finalizing_copy(self):
        with tempfile.TemporaryDirectory() as d:
            source,target=Path(d)/'source',Path(d)/'target';source.write_bytes(b'before')
            original=c.sha
            def mutate(p):
                result=original(p)
                if p==source:source.write_bytes(b'after')
                return result
            with patch.object(c,'sha',side_effect=mutate):
                with self.assertRaises(ValueError):c.checked_copy(source,target)
            self.assertFalse(target.exists())

    def test_cross_root_mapping_uses_registered_destination(self):
        registry={'worktrees':[{'worktree':'/old/one','archive':'/new/one'},
                               {'worktree':'/old/two','archive':'/new/two'}]}
        self.assertEqual(c.map_path('/old/two/runs/x',registry),Path('/new/two/runs/x'))
        with self.assertRaises(ValueError):c.map_path('/old/twosome/runs/x',registry)

    def test_broken_archived_link_blocks_validation(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT/'.scratch/archive-consolidation') as d:
            root=Path(d);(root/'link').symlink_to('missing')
            m=[{'archive':d,'links':[{'path':'link','destination_target':str(root/'missing')}]}]
            with patch.object(c,'manifests',return_value=m):
                with self.assertRaises(ValueError):c.verify_links()

    def test_removal_rejects_unregistered_main_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);allowed=root/'allowed';allowed.mkdir();alias=root/'alias';alias.symlink_to(allowed)
            r={'worktrees':[{'worktree':str(allowed)},{'worktree':str(c.ROOT)},{'worktree':str(alias)}]}
            self.assertEqual(c.removal_guard(allowed,r),allowed)
            for p in [c.ROOT,root,root/'unknown',alias]:
                with self.assertRaises(ValueError):c.removal_guard(p,r)


if __name__=='__main__':unittest.main()

class DeletionBoundaryTests(unittest.TestCase):
    def test_optimized_python_keeps_guard(self):
        import subprocess,sys
        result=subprocess.run([sys.executable,'-O','-c','from experiments.remove_archived_worktrees import require; require(False)'],capture_output=True)
        self.assertNotEqual(result.returncode,0)

    def test_backup_addition_and_destination_mutation_block_deletion(self):
        from experiments import remove_archived_worktrees as r
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);backup=root/'backup';backup.mkdir();source=backup/'x';source.write_text('data');dest=root/'saved';dest.write_text('data')
            audit={'passed':True,'files':[{'path':'x','identity':c.identity(source),'destination':'saved','destination_identity':c.identity(dest)}],'links':[]}
            with patch.object(r,'load',return_value=audit),patch.object(c,'ROOT',root):
                r.verify_backup_snapshot(backup)
                (backup/'new').write_text('unaccounted')
                with self.assertRaises(ValueError):r.verify_backup_snapshot(backup)
                (backup/'new').unlink();dest.write_text('tampered')
                with self.assertRaises(ValueError):r.verify_backup_snapshot(backup)

    def test_partial_backup_resume_requires_all_saved_destinations(self):
        from experiments import remove_archived_worktrees as r
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);backup=root/'backup';backup.mkdir();dest=root/'saved';dest.write_text('data')
            audit={'passed':True,'files':[{'path':'already-removed','identity':[],'destination':'saved','destination_identity':c.identity(dest)}],'links':[]}
            with patch.object(r,'load',return_value=audit),patch.object(c,'ROOT',root):
                with self.assertRaises(ValueError):r.verify_backup_snapshot(backup)
                r.verify_backup_snapshot(backup,partial=True)
                dest.unlink()
                with self.assertRaises(ValueError):r.verify_backup_snapshot(backup,partial=True)
