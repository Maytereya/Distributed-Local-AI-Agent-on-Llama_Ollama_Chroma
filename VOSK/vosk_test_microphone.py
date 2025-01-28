#!/usr/bin/env python3

import json
import os
import sys
import asyncio
import websockets
import logging
import sounddevice as sd
import argparse
import time


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))


async def run_test_base():
    with sd.RawInputStream(samplerate=args.samplerate, blocksize=4000, device=args.device, dtype='int16',
                           channels=1, callback=callback) as device:
        async with websockets.connect(args.uri) as websocket:
            await websocket.send('{ "config" : { "sample_rate" : %d } }' % device.samplerate)

            while True:
                data = await audio_queue.get()
                await websocket.send(data)
                print(await websocket.recv())

            await websocket.send('{"eof" : 1}')
            print(await websocket.recv())


async def run_test():
    final_text = []
    start_time = time.time()
    duration = 10  # продолжительность записи в секундах
    print("Начало 10-секундного раунда записи")

    with sd.RawInputStream(samplerate=args.samplerate, blocksize=4000, device=args.device, dtype='int16',
                           channels=1, callback=callback) as device:
        async with websockets.connect(args.uri) as websocket:
            await websocket.send('{ "config" : { "sample_rate" : %d } }' % device.samplerate)

            while time.time() - start_time < duration:
                data = await audio_queue.get()
                await websocket.send(data)
                result = await websocket.recv()
                # print(result)
                final_text.append(result)

            await websocket.send('{"eof" : 1}')
            final_result = await websocket.recv()
            # final_text.append(final_result)
            # print(final_result)

    # full_text = " ".join(final_text)
    print("\nИтоговый распознанный текст:")
    final_result_dict = json.loads(final_result)
    print(final_result_dict["text"])


async def main():
    global args
    global loop
    global audio_queue

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-l', '--list-devices', action='store_true',
                        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(description="ASR Server",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     parents=[parser])
    parser.add_argument('-u', '--uri', type=str, metavar='URL',
                        help='Server URL', default='ws://46.0.234.32:2700')
    parser.add_argument('-d', '--device', type=int_or_str,
                        help='input device (numeric ID or substring)')
    parser.add_argument('-r', '--samplerate', type=int, help='sampling rate', default=16000)
    args = parser.parse_args(remaining)
    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue()

    logging.basicConfig(level=logging.INFO)
    await run_test()


if __name__ == '__main__':
    asyncio.run(main())
