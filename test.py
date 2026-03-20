import asyncio
import sys
from bleak import BleakScanner, BleakClient
from bleak.exc import BleakError


async def scan_for_devices():
    """
    Scans for nearby Bluetooth Low Energy (BLE) devices.
    Returns a list of discovered devices.
    """
    print("-----------------------------------------")
    print("Scanning for nearby Bluetooth devices...")
    print("-----------------------------------------")

    # Scan for 5 seconds. You can adjust this duration.
    devices = await BleakScanner.discover(timeout=5.0)

    if not devices:
        print("No devices found. Make sure Bluetooth is on and devices are discoverable.")
        return []

    print(f"Found {len(devices)} device(s):")

    for i, device in enumerate(devices):
        # Some devices might not have a name
        name = device.name if device.name else "Unknown Device"
        print(f"[{i+1}] Name: {name}, Address: {device.address}, RSSI: {device.rssi}dBm")

    return devices


async def connect_and_pair(device_address):
    """
    Attempts to connect to a device by its address.
    On many operating systems, this connection attempt
    will trigger the OS's native pairing/bonding process.
    """
    print(f"\nAttempting to connect to {device_address}...")

    try:
        async with BleakClient(device_address) as client:
            print(f"Successfully connected to {device_address}!")

            # Check pairing status (not supported everywhere)
            try:
                is_paired = await client.is_paired()
                print(f"Pairing status: {'Paired/Bonded' if is_paired else 'Not Paired'}")
            except NotImplementedError:
                print("Pairing status check is not supported on this platform.")
                print("Connection success often implies pairing handled by OS.")

            # --- Optional: interact with device ---
            # services = await client.get_services()
            # print("Services discovered:")
            # for service in services:
            #     print(service)
            # --------------------------------------

            print("\nConnection will now be closed.")

    except BleakError as e:
        print(f"Failed to connect: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


async def main():
    """
    Main function to run the scanning and connection menu.
    """
    devices = await scan_for_devices()

    if not devices:
        return

    try:
        choice = int(input("\nEnter device number to connect (0 to exit): "))
        if choice == 0:
            return

        selected_device = devices[choice - 1]
        await connect_and_pair(selected_device.address)

    except (ValueError, IndexError):
        print("Invalid selection. Try again.")


if __name__ == "__main__":
    asyncio.run(main())