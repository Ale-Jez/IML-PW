from tensorboard.compat.proto import event_pb2
import struct

filepath = r'C:\Users\pczec\Desktop\Studia\SEM5\IML\IML-PW\logs\20260125-130124\events.out.tfevents.1769342484.Piotr-Legion5.20808.12'

with open(filepath, 'rb') as f:
    count = 0
    while True:
        try:
            length_bytes = f.read(8)
            if len(length_bytes) < 8: 
                break
            length = struct.unpack('<Q', length_bytes)[0]
            data = f.read(length)
            if len(data) < length: 
                break
            event = event_pb2.Event()
            event.ParseFromString(data)
            count += 1
            print(f'Event {count}: wall_time={event.wall_time}, step={event.step}, summary_values={len(event.summary.value)}')
            for v in event.summary.value:
                print(f'  Tag: {v.tag}')
            f.read(8)
        except Exception as e:
            print(f"Error: {e}")
            break
