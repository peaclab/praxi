# DeltaSherlock. See README.md for usage. See LICENSE for MIT/X11 license info.
"""
DeltaSherlock client watchdog module. Contains methods for analyzing the
filesystem and creating changesets.
"""
# pylint: disable=C0326, R0913
import time
from os import path
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
from deltasherlock.common.changesets import Changeset


class DeltaSherlockEventHandler(PatternMatchingEventHandler):
    """
    Default handler for filesystem events. Called on each file creation,
    modification, deletion, and move.
    """

    def __init__(self, changeset, patterns=None, ignore_patterns=None,
                 ignore_directories=True, case_sensitive=False):
        super(DeltaSherlockEventHandler, self).__init__(
            patterns, ignore_patterns, ignore_directories, case_sensitive)
        self.current_changeset = changeset

    def on_created(self, event):
        self.current_changeset.add_creation_record(event.src_path, time.time())

    def on_modified(self, event):
        self.current_changeset.add_modification_record(
            event.src_path, time.time())

    def on_deleted(self, event):
        self.current_changeset.add_deletion_record(event.src_path, time.time())

    def on_moved(self, event):
        # Treated as a deletion of the source and a creation of the destination
        self.current_changeset.add_deletion_record(event.src_path, time.time())
        self.current_changeset.add_creation_record(event.dest_path, time.time())

    def replace_changeset(self, new_changeset):
        """
        Swap out the current changeset being recorded to with a new changeset.
        :param new_changeset: the changeset to be "swapped in" to the watchdog
        :return: the old, closed changeset that you just replaced
        """
        if not new_changeset.open:
            raise ValueError("Cannot give a closed changeset to event handler")

        old_changeset = self.current_changeset
        self.current_changeset = new_changeset
        old_changeset.close(time.time())
        return old_changeset


class DeltaSherlockWatchdog(object):
    """
    Manages the watchdog that monitors the filesystem for changes
    """

    def __init__(self, paths: list, patterns: str = "*", ignore_patterns: str = None):
        """
        See http://pythonhosted.org/watchdog/api.html#watchdog.events.PatternMatchingEventHandler
        for explanation on the "patterns" parameters
        """
        # Create changeset infrastructure
        self.__changesets = []

        self.__observer = Observer()
        self.__handler = DeltaSherlockEventHandler(Changeset(time.time()),
                                                   patterns=patterns,
                                                   ignore_patterns=ignore_patterns,
                                                   ignore_directories=True,
                                                   case_sensitive=False)
        for p in paths:
            if path.isfile(p) or path.isdir(p):
                self.__observer.schedule(self.__handler, p, recursive=True)
            else:
                # Path does not exist.
                # TODO: throw warning?
                pass
        self.__observer.start()
        return

    def __del__(self):
        self.__observer.stop()
        # Block until thread has stopped
        self.__observer.join()
        return

    def mark(self) -> Changeset:
        """
        Close the current changeset being recorded to, open a new one, and
        return the former
        :return: the old, closed changeset that was just "ejected"
        """
        latest_changeset = self.__handler.replace_changeset(
            Changeset(time.time()))
        self.__changesets.append(latest_changeset)
        return latest_changeset

    def get_changeset(self, first_index: int, last_index: int = None) -> Changeset:
        """
        Returns the sum of all changesets between two indexes (inclusive of
        first, exclusive of last), or just the single changeset specified
        :param first_index: the index of the first changeset you'd like to
        include
        :param last_index: the index of the changeset AFTER the last changeset
        you'd like to include, or None if you only want the changeset specified
        by first_index
        :return: the sum of all changesets across the specified range
        """
        sum_changeset = self.__changesets[first_index]

        if last_index is not None:
            for changeset in self.__changesets[first_index + 1:last_index]:
                sum_changeset += changeset

        return sum_changeset
