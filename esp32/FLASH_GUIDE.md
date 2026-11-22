# Complete guide: upload CoverageAI and its ML model to ESP32

## Important: the model is compiled into the firmware

You do **not** upload a `.pkl` or `.csv` model file separately to ESP32 flash.
The model is exported as `wifi_rssi_ridge.h`. Arduino compiles that header and
the `.ino` sketch into one firmware image. Pressing **Upload** writes that
firmware image, including the model, to the ESP32.

Use this Arduino sketch folder:

```text
COVERAGE_AI_ML/
  esp32/
    COVERAGE_AI/
      COVERAGE_AI.ino
      wifi_rssi_ridge.h
      wifi_rssi_calibration.h
```

Keep all three files together. Open the `.ino` inside the `COVERAGE_AI` folder,
not the older `esp32/COVERAGE_AI.ino` file. Do not rename the sketch folder or
the `.ino` to different names; Arduino expects them to match.

## 1. Prepare the ESP32 and computer

1. Connect the ESP32 to the computer with a USB **data** cable. A charge-only
   cable will power the board but will not provide an upload port.
2. Install [Arduino IDE 2](https://www.arduino.cc/en/software/) from Arduino.
3. In Arduino IDE, open **File → Preferences**. In **Additional boards manager
   URLs**, add Espressif's stable package URL if it is not already listed:
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`.
   Espressif's [Arduino-ESP32 installation guide](https://docs.espressif.com/projects/arduino-esp32/en/latest/installing.html)
   has screenshots and alternate installation methods.
4. Open **Tools → Board → Boards Manager**, search for `esp32`, and install
   **esp32 by Espressif Systems**.
5. Open **Tools → Manage Libraries** and install:
   - TinyGPSPlus
   - Adafruit GFX Library
   - Adafruit SSD1306
   - Adafruit BMP280 Library

   Accept dependency installation prompts. `Wire`, `SPI`, `SD`, `WiFi`,
   `WebServer`, and `DNSServer` come with the ESP32 board package.

## 2. Open the correct firmware files

1. In File Explorer, open your project folder, then
   `COVERAGE_AI_ML\esp32\COVERAGE_AI`.
2. Confirm that the folder contains exactly the sketch and the two headers
   listed above.
3. Before uploading, open `COVERAGE_AI.ino` in a text editor or Arduino IDE and
   change this line to a private Wi-Fi password of at least 8 characters:

   ```cpp
   const char *AP_PASS = "coverage123";
   ```

4. Save the sketch. The model header stays beside the `.ino`; it is included
   automatically during compilation. You do not need to add it as a separate
   Arduino tab or copy the model to the SD card.

## 3. Check board and wiring settings

1. In Arduino IDE, choose **Tools → Board** and select the board that matches
   the physical ESP32. For many ESP32-WROOM development boards, choose
   **ESP32 Dev Module**. For a TTGO T-Call, use the matching board entry if it
   appears in the list. The exact menu options vary by board package version.
2. Choose **Tools → Port** and select the COM port that appears when the board
   is connected. If no port appears, install the board's USB-to-serial driver,
   try a data cable, and reconnect the board.
3. Check the sketch's pin definitions against the wiring before powering
   attached peripherals:

   | Peripheral | Sketch setting |
   |---|---|
   | I2C SDA / SCL | GPIO 21 / GPIO 22 |
   | GPS receive | GPIO 34, UART at 9600 baud |
   | SD CS / SCK / MISO / MOSI | GPIO 14 / 18 / 19 / 25 |
   | OLED | I2C address `0x3C` |
   | MPU6500 | I2C address `0x68` |
   | BMP280 | I2C address `0x76` |

   Verify supply voltage and signal levels for your board and modules. Do not
   assume every ESP32 or add-on board uses these pins.

## 4. Compile and upload

1. Click **Verify** (the checkmark). Wait for `Done compiling`.
2. If compilation reports a missing library, install that library using
   **Tools → Manage Libraries**, then click Verify again.
3. Click **Upload** (the right arrow). Keep the board connected while the IDE
   compiles and transfers the firmware.
4. If the IDE remains at `Connecting...`, hold the ESP32 **BOOT** button. If
   needed, tap **EN/RESET** while holding BOOT; release BOOT once upload begins.
5. Wait for the IDE to report `Done uploading`, then open **Tools → Serial
   Monitor** and set the baud rate to **115200**. Press EN/RESET once if the
   startup messages do not appear.
6. Look for initialization lines for Wi-Fi, SD, display, GPS and sensors. A
   missing optional peripheral can report an error while the ESP32 firmware
   and dashboard still start; an SD error means logs will not be saved.

## 5. Connect and use the dashboard

1. On a phone or computer, join **CoverageAI** using the password set in
   `AP_PASS`.
2. Open `http://192.168.4.1`. If the captive portal does not appear, enter the
   address directly.
3. Press **Start Survey** to record the scan log. Pause, Resume, Stop and New
   Survey control logging; the RSSI-only map itself does not use survey
   coordinates, GPS, pressure, temperature or motion sensors.
4. The square heatmap is built from the latest Wi-Fi RSSI scan. Every detected
   AP gets its own tappable dot. The dot's radial distance from YOU is estimated
   from RSSI using `RSSI_at_1m = -40 dBm` and path-loss exponent `n = 2.2`.
   AP direction cannot be measured from RSSI, so the map assigns a stable,
   schematic direction from each BSSID. The colored surface models signal
   falloff from those distance anchors and shows the strongest predicted RSSI
   at each cell. It is an estimate, not a floorplan or surveyed AP location.
5. Tap an AP dot to see its RSSI, predicted device distance and estimated
   `-70 dBm` range. The pairwise panel gives only a possible interval between
   two APs from their device-distance estimates; RSSI alone cannot determine
   exact AP-to-AP spacing or bearing. Drag to pan and use the wheel or zoom
   buttons. When the strongest current AP is below `-70 dBm`, the dashboard
   searches the heatmap grid for a suggested repeater spot. It excludes cells
   within 2 m of schematic AP markers, simulates a repeater at `-50 dBm` at 1 m,
   and displays weak-cell percentage before and after. This is a model estimate;
   RSSI-only AP positions cannot prove that a real router is absent or guarantee
   actual improvement.
6. **Load survey history** lists SD sessions. `/csv` downloads all SD history
   and `/csv?session=ID` downloads one session. The card file is
   `/coverage_ai_survey.csv`.

## 6. RSSI estimate limits

RSSI-based distance is approximate. Walls, antenna direction, people,
interference, transmit power, multipath and the assumed path-loss exponent can
change it substantially. The map uses only RSSI for its distances and signal
surface; it does not claim to recover AP bearing or physical transmitter
coordinates. `-70 dBm` range is calculated using the same reference and path
loss assumptions.

## 7. Calibrate the RSSI distance assumptions

The firmware starts with `RSSI@1m = -40 dBm` and path-loss exponent `n = 2.2`.
These are defaults, not measured values for your router. To calibrate:

1. Keep the ESP32 in the same orientation and height. Use one router and record
   its RSSI at measured distances of 1 m, 3 m, 5 m and 8 m. Collect about 10
   scans per distance and use the median RSSI for each distance.
2. In a spreadsheet, make columns `x = LOG10(distance_m)` and `y = median RSSI`.
   Fit `y = A + B*x`; set `RSSI@1m` to `A` and `n` to `-B/10`.
3. Update `DEFAULT_RSSI_AT_1M_DBM` and `DEFAULT_PATH_LOSS_EXPONENT` in the sketch,
   upload again, and repeat the distance measurements at separate points.
4. Compare predicted distance with tape-measured distance and report median
   absolute error in metres. Keep the same router, band, ESP32 orientation and
   room conditions for the comparison.

Use calibrated measurements in the presentation only after collecting them.
The repeater simulation reads `n` from the firmware heatmap response so it uses
the same path-loss exponent as the range estimate.

## 8. Update the ML model later

The checked-in Ridge model is a public Tampere dataset benchmark. It is not
trained for your router, building or ESP32 radio. To create a site model, first
collect scans at multiple known coordinates, floors and buildings. Then in a
terminal opened at `COVERAGE_AI_ML`, run:

```powershell
python src\ingest_esp32_csv.py path\to\coverage_ai_survey.csv
python src\train_models.py --dataset ESP32 --full
python src\export_ridge_esp32.py
python src\validate_ridge_export.py
```

Review the model evaluation and export validation. The export script places
`wifi_rssi_ridge.h` in `esp32/` and in `esp32/COVERAGE_AI/`, beside the Arduino
sketch. Reopen the sketch in Arduino IDE, click **Verify**, then **Upload**.
That new upload writes the new compiled model into the ESP32's program flash.
The board does not change model until this firmware upload completes.

Do not copy `ridge_model.pkl`, `model_bundle.pkl`, training CSVs, or Python
files to the ESP32. Those are for the computer-side training pipeline. The
exported C++ header is the model representation used by the firmware.

`calibrate_esp32.py` can estimate a constant offset against held-out field
locations for the public benchmark; it does not make the benchmark a site
model. Exporting a separately trained site model resets that benchmark offset.

## Troubleshooting

- **`wifi_rssi_ridge.h: No such file`**: verify that the header is in the same
  `COVERAGE_AI` folder as `COVERAGE_AI.ino`, then reopen that sketch.
- **Missing TinyGPSPlus or Adafruit header**: install the named library and its
  dependencies in Library Manager.
- **No COM port**: try a known data cable and another USB port; install the
  USB-serial driver required by the board.
- **Upload timeout**: close Serial Monitor, select the correct port, retry,
  then use the BOOT/EN sequence described above.
- **Wi-Fi connects but page does not open**: enter `http://192.168.4.1`
  manually and check Serial Monitor for `softAP` and web-server startup.
- **No heatmap or repeater estimate**: confirm Wi-Fi scans are updating, then
  reload the dashboard. A repeater suggestion appears only when the strongest
  detected AP is below `-70 dBm` and a grid candidate improves the modeled weak
  cell percentage while remaining at least 2 m from schematic AP markers.
- **No SD file**: check card format, wiring, CS pin, and the SD initialization
  message in Serial Monitor.

## Verification limits

The embedded dashboard JavaScript and exported C++ Ridge predictions were
validated in this project. The firmware still needs to be compiled and tested
on the specific ESP32 board and attached sensors. Check RAM/flash use and
confirm predictions against local measurements before using the results for a
deployment decision.
