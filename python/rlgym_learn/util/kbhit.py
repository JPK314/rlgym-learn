# pyright: reportPossiblyUnboundVariable=false

#!/usr/bin/env python
"""
A Python class implementing KBHIT, the standard keyboard-interrupt poller.
Works transparently on Windows and Posix (Linux, Mac OS X).  Doesn't work
with IDLE.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""

import os

# Windows
if os.name == "nt":
    import msvcrt

# Posix (Linux, OS X)
else:
    import atexit
    import sys
    import termios
    from select import select


class KBHit:
    def __init__(self):
        """Creates a KBHit object that you can call to do various keyboard things."""

        if os.name == "nt":
            pass

        else:
            # Save the terminal settings
            self.fd: int = sys.stdin.fileno()
            self.new_term: termios._AttrReturn = termios.tcgetattr(self.fd)  # pyright: ignore [reportPrivateUsage]
            self.old_term: termios._AttrReturn = termios.tcgetattr(self.fd)  # pyright: ignore [reportPrivateUsage]

            # New terminal setting unbuffered
            self.new_term[3] = self.new_term[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

            # Support normal-terminal reset at exit
            _ = atexit.register(self.set_normal_term)

    def set_normal_term(self):
        """Resets to normal terminal.  On Windows this is a no-op."""

        if os.name == "nt":
            pass

        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def getch(self):
        """Returns a keyboard character after kbhit() has been called.
        Should not be called in the same program as getarrow().
        """

        if os.name == "nt":
            return msvcrt.getch().decode("utf-8")

        else:
            return sys.stdin.read(1)

    def kbhit(self):
        """Returns True if keyboard character was hit, False otherwise."""
        if os.name == "nt":
            return msvcrt.kbhit()

        else:
            dr, _, _ = select([sys.stdin], [], [], 0)
            return dr != []


# Test
if __name__ == "__main__":
    kb = KBHit()

    print("Hit any key, or ESC to exit")

    while True:
        if kb.kbhit():
            c = kb.getch()
            if ord(c) == 27:  # ESC
                break
            print(c)

    kb.set_normal_term()
