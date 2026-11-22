#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <WiFi.h>
#include <WebServer.h>
#include <DNSServer.h>
#include <stdarg.h>
#include <HardwareSerial.h>
#include <esp_system.h>
#include <TinyGPSPlus.h>
#include <math.h>

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_BMP280.h>
#if __has_include("wifi_rssi_ridge.h")
#include "wifi_rssi_ridge.h"
#define WIFI_RIDGE_MODEL_AVAILABLE 1
#else
#define WIFI_RIDGE_MODEL_AVAILABLE 0
namespace wifi_ridge {
inline double ap_mean_rssi(const char *) { return -75.0; }
inline double predictRSSI(double, double, int, int, double) { return -100.0; }
inline int coverageClass(double r) { return r >= -50 ? 5 : r >= -60 ? 4 : r >= -70 ? 3 : r >= -80 ? 2 : 1; }
}
#endif
#if __has_include("wifi_rssi_calibration.h")
#include "wifi_rssi_calibration.h"
#define WIFI_RSSI_CALIBRATION_AVAILABLE 1
#else
#define WIFI_RSSI_CALIBRATION_AVAILABLE 0
namespace wifi_calibration { static const double RSSI_OFFSET_DBM = 0.0; }
#endif

// =====================================================
// COVERAGE-AI INTEGRATED DATA COLLECTION SYSTEM  (+ WEBSITE)
// ESP32 + GPS + Wi-Fi + MPU6500 + BMP280 + OLED + SD
//
// Scans every 15 s, logs to SD, reads sensors, serves the dashboard, and runs
// Ridge RSSI inference locally on the ESP32. No laptop server is required.
// Added:
//   * ESP32 hosts its own Wi-Fi network + website
//       join "CoverageAI"  ->  open http://192.168.4.1
//   * Live page: status, sensors, AP list, measured and ML-predicted heatmaps
//   * Relative survey coordinates begin at (0,0); GPS is optional
//   * Survey controls and BSSID-isolated measured / IDW / optional ML maps
//   * Session-aware SD CSV history; RAM holds only recent map samples
//   * Timestamped serial logs  [ms][TAG] message
// =====================================================

// ---------------- WEBSITE SETTINGS ----------------
const char *AP_SSID = "CoverageAI";     // network the ESP32 creates
const char *AP_PASS = "coverage123";    // min 8 characters
const char *TARGET_SSID = "";           // router to map ("" = strongest AP)
const int DEFAULT_FLOOR_ID = 0;
const int DEFAULT_BUILDING_NUM = 0;
const double IDW_MAX_DISTANCE_M = 12.0;
const double DEFAULT_PATH_LOSS_EXPONENT = 2.2;
const double DEFAULT_RSSI_AT_1M_DBM = -40.0;
const double USEFUL_SIGNAL_THRESHOLD_DBM = -70.0;
#define LOG_HTTP_POLLS 0                // 1 = also log the page's /data polling

// ---------------- I2C ----------------
#define I2C_SDA 21
#define I2C_SCL 22

// ---------------- OLED ----------------
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDRESS 0x3C

Adafruit_SSD1306 display(
  SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1
);

// ---------------- GPS ----------------
HardwareSerial GPS(1);
TinyGPSPlus gps;

#define GPS_RX 34

// ---------------- BMP280 ----------------
Adafruit_BMP280 bmp;

// ---------------- MPU6500 ----------------
#define MPU_ADDR 0x68
#define MPU_WHO_AM_I 0x75
#define MPU_PWR_MGMT_1 0x6B
#define MPU_ACCEL_CONFIG 0x1C
#define MPU_GYRO_CONFIG 0x1B
#define MPU_ACCEL_XOUT_H 0x3B

#define ACCEL_SCALE 4096.0f
#define GYRO_SCALE 65.5f
#define GRAVITY 9.80665f

// ---------------- MICROSD ----------------
#define SD_CS 14
#define SD_SCK 18
#define SD_MISO 19
#define SD_MOSI 25

SPIClass sdSPI(VSPI);

// Versioned name prevents mixing the common schema with older CSV layouts.
const char *filename = "/coverage_ai_survey.csv";

// ---------------- WEB SERVER ----------------
WebServer server(80);

// Captive portal: answers EVERY domain lookup with the ESP32's own IP,
// so phones detect a "sign-in" network and pop the page up automatically.
DNSServer dnsServer;
const byte DNS_PORT = 53;

// ---------------- STATUS ----------------
bool oledOK = false;
bool gpsOK = false;
bool mpuOK = false;
bool bmpOK = false;
bool sdOK = false;

bool scanInProgress = false;

unsigned long lastSensorRead = 0;
unsigned long lastOLEDUpdate = 0;
unsigned long nextScanTime = 0;
unsigned long lastHeartbeat = 0;

const unsigned long sensorInterval = 1000;
const unsigned long oledInterval = 1000;
const unsigned long scanInterval = 15000;
const unsigned long heartbeatInterval = 10000;

// ---------------- SENSOR DATA ----------------
float ax = 0, ay = 0, az = 0;
float gx = 0, gy = 0, gz = 0;

float temperatureC = 0;
float pressureHpa = 0;

int networksFound = 0;
int strongestRSSI = -127;

String strongestSSID = "";

// ---------------- WEB / LOG DATA ----------------
struct LiveAp {
  char ssid[33];
  char bssid[18];
  int rssi;
  int ch;
  bool open;
  double apMean;
  float predicted;
  int coverage;
  bool predictionValid;
  uint32_t seenScan;
  double distanceM;
  double usefulRangeM;
};
const int MAX_LIVE_APS = 30;
LiveAp liveAps[MAX_LIVE_APS];
int liveApCount = 0;

struct Sample {
  uint32_t measurementID;
  uint32_t timestamp;
  float x, y;
  int8_t rssi;
  uint8_t channel;
  float predicted;
  bool gpsValid;
  char ssid[33];
  char bssid[18];
};
const int MAX_SAMPLES = 500;
Sample samples[MAX_SAMPLES];
int sampleCount = 0;
int sampleHead = 0;

unsigned long scanCounter = 0;
unsigned long scanStartMs = 0;
unsigned long sdRows = 0;
unsigned long sdFails = 0;
bool lastFixState = false;

// ---------------- PHONE GPS FALLBACK ----------------
// Used only when the GPS module has no fix. The phone shares its location
// through the website (/loc) while it is connected to the ESP32's Wi-Fi.
const unsigned long PHONE_MAX_AGE_MS = 15000;  // ignore phone fixes older than this
const float PHONE_MAX_ACC_M = 50.0f;           // ignore phone fixes worse than this
double phoneLat = 0, phoneLon = 0;
float phoneAcc = 9999;
unsigned long phoneFixMs = 0;
bool phoneEver = false;
int lastPosSrc = 0;

// GPS origin and optional manually-entered local position for indoor surveys.
bool originSet = false;
double originLat = 0.0, originLon = 0.0;
double gpsAnchorX = 0.0, gpsAnchorY = 0.0;
bool gpsAuto = false;
double manualX = 0.0, manualY = 0.0;
int siteFloorId = DEFAULT_FLOOR_ID;
int siteBuildingNum = DEFAULT_BUILDING_NUM;
bool surveyStarted = false;
bool surveyPaused = false;
bool surveyStopped = false;
String currentSessionId;
uint32_t surveyMeasurementCount = 0;
bool surveyHasBounds = false;
double surveyMinX = 0.0, surveyMaxX = 0.0, surveyMinY = 0.0, surveyMaxY = 0.0;

// Device-learned AP means replace the anonymized public table as observations
// arrive. The exported model table remains a fallback for APs not yet seen.
struct ApRunningMean {
  char bssid[18];
  double sum;
  uint16_t count;
};
const int MAX_AP_STATS = 64;
ApRunningMean apStats[MAX_AP_STATS];
int apStatsCount = 0;

const double pathLossExponent = DEFAULT_PATH_LOSS_EXPONENT;

// =====================================================
// LOGGING   [uptime ms][TAG] message
// =====================================================

void logMsg(const char *tag, const char *fmt, ...) {

  char buf[220];

  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);

  Serial.printf("[%8lu][%-5s] %s\n", millis(), tag, buf);
}

void onWiFiEvent(WiFiEvent_t event) {

  switch (event) {

    case ARDUINO_EVENT_WIFI_AP_START:
      logMsg("WIFI", "Access point started");
      break;

    case ARDUINO_EVENT_WIFI_AP_STACONNECTED:
      logMsg("WIFI", "Phone joined the ESP32 network (clients: %d)",
             WiFi.softAPgetStationNum());
      break;

    case ARDUINO_EVENT_WIFI_AP_STADISCONNECTED:
      logMsg("WIFI", "Phone left the ESP32 network (clients: %d)",
             WiFi.softAPgetStationNum());
      break;

    case ARDUINO_EVENT_WIFI_AP_STAIPASSIGNED:
      logMsg("WIFI", "Phone received an IP address");
      break;

    default:
      break;
  }
}

// =====================================================
// WEB PAGE (stored in flash)
// =====================================================

const char PAGE[] PROGMEM =
  "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
  "<title>COVERAGE-AI Survey</title><style>\n"
  ":root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#eef4ff;font:14px system-ui, sans-serif}.wrap{max-width:940px;margin:auto;padding:14px}h1{font-size:22px;margin:2px 0 12px}h2{font-size:15px;color:#a9c8ff;margin:14px 0 8px}.row,.metrics,.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.metrics{margin:8px 0}.card{background:#14213a;border:1px solid #243657;border-radius:12px;padding:10px}.metric{flex:1;min-width:115px}.metric b{font-size:19px;display:block}.controls{justify-content:center;margin:8px 0}button,select,input{border:1px solid #35517c;border-radius:7px;background:#172944;color:#fff;padding:8px}button{cursor:pointer}button:hover{background:#244878}.primary{background:#126b42}.danger{background:#8b3030}.muted{color:#a9bddc;font-size:12px}#map{display:block;width:min(100%,620px);height:auto;aspect-ratio:1;margin:8px auto;border:1px solid #436187;border-radius:8px;touch-action:none;background:#101a2d}.legend{display:flex;justify-content:center;gap:10px;flex-wrap:wrap;font-size:12px}.sw{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:-2px;margin-right:3px}.split{display:flex;height:8px;border-radius:6px;overflow:hidden;margin-top:8px;background:#101a2d}.split div{height:100%}table{width:100%;border-collapse:collapse;font-size:12px}td,th{text-align:left;border-bottom:1px solid #283b5b;padding:6px}a{color:#ffc66d}#message{min-height:18px;text-align:center;color:#ffc66d}.pad{display:grid;grid-template-columns:repeat(3,46px);gap:4px;justify-content:center}.pad button{height:42px;font-size:18px}.spacer{visibility:hidden}label{display:inline-flex;gap:5px;align-items:center}\n"
  "</style></head><body><main class=\"wrap\"><h1>COVERAGE-AI <span class=\"muted\">Wi-Fi survey</span></h1>\n"
  "<div class=\"metrics\"><div class=\"card metric\">Survey<b id=\"state\">Ready</b></div><div class=\"card metric\">Networks<b id=\"nets\">0</b></div><div class=\"card metric\">Measurements<b id=\"measurements\">0</b></div><div class=\"card metric\">Map basis<b id=\"position\">RSSI only</b></div><div class=\"card metric\">SD<b id=\"sd\">Waiting</b></div></div>\n"
  "<section class=\"card\"><h2>Survey controls</h2><div class=\"controls\"><button class=\"primary\" onclick=\"action('/survey/start')\">Start Survey</button><button onclick=\"action('/survey/pause')\">Pause</button><button onclick=\"action('/survey/resume')\">Resume</button><button class=\"danger\" onclick=\"action('/survey/stop')\">Stop</button><button onclick=\"action('/survey/clear')\">New Survey</button></div><div class=\"muted\" style=\"text-align:center\">Session <span id=\"session\">-</span> \302\267 scans continue without GPS; records are added only while a survey is active.</div></section>\n"
  "<section><h2>Wi-Fi heatmap <span id=\"mapinfo\"></span></h2><div class=\"controls\"><span class=\"muted\">RSSI-only distance prediction</span><button onclick=\"zoom(1.25)\">Zoom +</button><button onclick=\"zoom(.8)\">Zoom \342\210\222</button><button onclick=\"resetView()\">Reset view</button></div><canvas id=\"map\" width=\"500\" height=\"500\"></canvas><div class=\"legend\"><span style=\"color:#238b45\">Strong >= -62 dBm</span><span style=\"color:#d5d953\">Fair -63 to -75 dBm</span><span style=\"color:#df4a3d\">Weak &lt; -75 dBm</span><span>No data = blank</span><span>AP dots: tap for RSSI and range details</span><span>AP dot angles are schematic; RSSI has no bearing</span></div><div class=\"split\" id=\"split\"></div><p class=\"muted\" id=\"splitTxt\"></p><p class=\"muted\" id=\"mapnote\" style=\"text-align:center\">YOU is centered. Heatmap and AP distance estimates use detected RSSI only. AP directions are schematic because RSSI cannot provide bearing.</p><div class=\"card\" id=\"inspect\">Tap/click the map to inspect a location.</div></section>\n"
  "\n"
  "<section class=\"card\"><h2>Wi-Fi coverage and repeater estimate</h2><div class=\"muted\">Distance assumes -40 dBm at 1 m and the firmware path-loss exponent (default n=2.2). When strongest RSSI is below -70 dBm, the grid estimates a repeater spot using -50 dBm at 1 m and the same exponent. AP bearings and candidate exclusion use schematic AP dots; verify on site.</div><div id=\"routerRec\" class=\"card\" style=\"margin:12px 0;border:1px solid #f59e0b\">Waiting for RSSI scan before recommending router placement.</div><div id=\"networkSummary\" class=\"muted\">Waiting for Wi-Fi scans.</div><h2>Pairwise AP distance</h2><div class=\"muted\">RSSI ranges only give a possible interval between two APs; exact spacing and bearing are unknown.</div><div id=\"pairwise\" class=\"muted\">Waiting for AP readings.</div></section>\n"
  "<div class=\"controls\"><a href=\"/csv\">Download complete SD survey CSV</a><button onclick=\"loadHistory()\">Load survey history</button><select id=\"history\"><option value=\"\">Select session</option></select><a id=\"sessionDownload\" href=\"/csv\">Download selected session</a><span id=\"gpsStatus\" class=\"muted\"></span></div><div id=\"message\"></div>\n"
  "</main><script>\n"
  "const cv=document.getElementById('map'),g=cv.getContext('2d');const state={samples:[],allPoints:[],aps:[],markers:[],apWorld:[],ml:null,repeaterPlan:null,model:false,x:0,y:0,bounds:null,view:null,drag:null,key:''};const MAX_DIST=12;\n"
  "function color(v){const stops=[[-100,[143,24,43]],[-85,[223,74,61]],[-75,[245,163,56]],[-67,[213,217,83]],[-60,[104,189,85]],[-50,[35,139,69]],[-30,[12,91,53]]];v=Math.max(-100,Math.min(-30,v));let i=0;while(i<stops.length-2&&v>stops[i+1][0])i++;const [a,ca]=stops[i],[b,cb]=stops[i+1],t=(v-a)/(b-a),rgb=ca.map((x,k)=>Math.round(x+(cb[k]-x)*t));return 'rgb('+rgb.join(',')+')'}\n"
  "function apColor(b){let n=0;for(let i=0;i<b.length;i++)n=(n*31+b.charCodeAt(i))>>>0;return 'hsl('+(n%360)+' 80% 62%)'}\n"
  "function renderCoverage(values){let strong=0,fair=0,weak=0,blank=0;for(const v of values){if(v===null||!Number.isFinite(Number(v))){blank++;continue}if(v>=-62)strong++;else if(v>=-75)fair++;else weak++}const total=values.length||1;document.getElementById('split').innerHTML=[[strong,'#22c55e'],[fair,'#fbbf24'],[weak,'#ef4444'],[blank,'#45546c']].map(x=>'<div style=\"width:'+100*x[0]/total+'%;background:'+x[1]+'\"></div>').join('');document.getElementById('splitTxt').textContent=values.length?'Strong '+Math.round(100*strong/total)+'% \302\267 Fair '+Math.round(100*fair/total)+'% \302\267 Weak '+Math.round(100*weak/total)+'% \302\267 No measured/modelled data '+Math.round(100*blank/total)+'% of map cells':'Waiting for actual RSSI samples';document.getElementById('mapinfo').textContent=values.length?(total-blank)+'/'+total+' cells mapped':''}\n"
  "function esc(s){return String(s).replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]))}\n"
  "function msg(s){document.getElementById('message').textContent=s;setTimeout(()=>{if(document.getElementById('message').textContent===s)document.getElementById('message').textContent=''},3500)}\n"
  "async function action(path){try{const r=await fetch(path);const t=await r.text();if(!r.ok)throw Error(t||r.status);let message=t;try{message=JSON.parse(t).message||t}catch(_){}if(path==='/survey/start'||path==='/survey/clear'){state.view=null;state.key=''}msg(message||'Done');await poll()}catch(e){msg('Request failed: '+e.message)}}\n"
  "async function loadHistory(){try{const r=await fetch('/survey/history'),a=await r.json(),s=document.getElementById('history');s.innerHTML='<option value=\"\">Select session</option>'+a.map(x=>'<option value=\"'+esc(x.session)+'\">'+esc(x.session)+' ('+x.measurements+')</option>').join('');s.onchange=()=>{document.getElementById('sessionDownload').href=s.value?'/csv?session='+encodeURIComponent(s.value):'/csv'};msg('Loaded '+a.length+' recent survey sessions')}catch(e){msg('Could not read survey history: '+e.message)}}\n"
  "function autoBounds(){let radius=10;for(const a of state.aps)if(a.seen)radius=Math.max(radius,Number(a.dist||0)+8);return {cx:0,cy:0,side:2*radius}}\n"
  "function transform(){const base=autoBounds(),v=state.view||{cx:0,cy:0,z:1};const side=base.side/v.z,scale=(cv.width-54)/side;return {v,side,scale,px:x=>cv.width/2+x*scale,py:y=>cv.height/2-y*scale,wx:x=>(x-cv.width/2)/scale,wy:y=>(cv.height/2-y)/scale}}\n"
  "function idw(x,y,P){const groups=new Map();for(const p of P){const k=p[6]||'';if(!groups.has(k))groups.set(k,[]);groups.get(k).push(p)}let best=null;for(const points of groups.values()){let sw=0,sv=0,near=1e9,nearRSSI=null,exact=null,count=0;for(const p of points){const d=Math.hypot(x-p[0],y-p[1]);if(d<near){near=d;nearRSSI=p[2]}if(d<MAX_DIST)count++;if(d<.03){exact=p[2];continue}if(d<=MAX_DIST){const w=1/(d*d+.0001);sw+=w;sv+=w*p[2]}}const v=exact!==null?exact:(sw?sv/sw:null);if(v!==null&&(!best||v>best.v))best={v,near,nearRSSI,count,measured:exact!==null}}return best}\n"
  "function weakCellPercent(values){let total=0,weak=0;for(const v of values){if(v===null||!Number.isFinite(Number(v)))continue;total++;if(Number(v)<-70)weak++}return total?100*weak/total:0}\n"
  "function planRepeater(h,apWorld){const vals=h.values,nx=h.nx,ny=h.ny,finite=[];for(let iy=0;iy<ny;iy++)for(let ix=0;ix<nx;ix++){const i=iy*nx+ix,v=vals[i];if(v!==null&&Number.isFinite(Number(v)))finite.push({i,ix,iy,v:Number(v),x:h.xmin+(h.xmax-h.xmin)*ix/(nx-1),y:h.ymin+(h.ymax-h.ymin)*iy/(ny-1)})}if(!finite.length)return null;const before=weakCellPercent(vals),n=Number(h.pathLossExponent)||2.2,ref=-50,threshold=-70;let best=null;for(const c of finite){let blocked=false;for(const ap of apWorld)if(Math.hypot(c.x-ap.x,c.y-ap.y)<2){blocked=true;break}if(blocked)continue;let weak=0,gain=0;for(const cell of finite){const d=Math.max(1,Math.hypot(cell.x-c.x,cell.y-c.y)),rep=ref-10*n*Math.log10(d),combined=Math.max(cell.v,rep);if(combined<threshold)weak++;gain+=combined-cell.v}const after=100*weak/finite.length;if(!best||after<best.after-1e-9||(Math.abs(after-best.after)<1e-9&&gain>best.gain))best={x:c.x,y:c.y,before,after,gain,n,clearance:apWorld.reduce((m,a)=>Math.min(m,Math.hypot(c.x-a.x,c.y-a.y)),Infinity)}}return best&&best.after<before?best:null}\n"
  "function draw(){state.markers=[];state.apWorld=[];const t=transform();g.clearRect(0,0,cv.width,cv.height);g.fillStyle='#101a2d';g.fillRect(0,0,cv.width,cv.height);if(state.ml){const h=state.ml,off=document.createElement('canvas');off.width=h.nx;off.height=h.ny;const og=off.getContext('2d');for(let iy=0;iy<h.ny;iy++)for(let ix=0;ix<h.nx;ix++){const val=h.values[iy*h.nx+ix];if(val===null)continue;og.fillStyle=color(val);og.fillRect(ix,h.ny-1-iy,1,1)}g.globalAlpha=.86;g.imageSmoothingEnabled=true;g.drawImage(off,t.px(h.xmin),t.py(h.ymax),t.scale*(h.xmax-h.xmin),t.scale*(h.ymax-h.ymin));g.globalAlpha=1;renderCoverage(h.values)}const step=t.side/10;g.strokeStyle='#22334c';g.lineWidth=1;for(let i=-5;i<=5;i+=2){const q=i*step;g.beginPath();g.moveTo(t.px(q),26);g.lineTo(t.px(q),cv.height-28);g.stroke();g.beginPath();g.moveTo(26,t.py(q));g.lineTo(cv.width-26,t.py(q));g.stroke()}g.fillStyle='#fff';g.beginPath();g.arc(cv.width/2,cv.height/2,7,0,Math.PI*2);g.fill();g.fillStyle='#ff4d6d';g.beginPath();g.arc(cv.width/2,cv.height/2,4,0,Math.PI*2);g.fill();g.textAlign='center';g.fillText('YOU',cv.width/2,cv.height/2-11);const aps=state.aps.filter(a=>a.seen);for(let i=0;i<aps.length;i++){const a=aps[i],range=Number(a.dist)||0,angle=(rangeBearing(a.b)%360)*Math.PI/180,wx=range*Math.cos(angle),wy=range*Math.sin(angle),x=t.px(wx),y=t.py(wy);state.apWorld.push({x:wx,y:wy});g.fillStyle=apColor(a.b);g.beginPath();g.arc(x,y,7,0,Math.PI*2);g.fill();g.strokeStyle='#fff';g.lineWidth=1;g.stroke();g.fillStyle='#101a2d';g.font='bold 9px sans-serif';g.fillText('AP'+(i+1),x,y+3);state.markers.push({x,y,ap:a,index:i+1,source:'RSSI-predicted distance '+range.toFixed(1)+' m; angle is schematic'})}const strongest=aps.reduce((m,a)=>Math.max(m,Number(a.r)),-Infinity),triggered=aps.length>0&&strongest< -70,plan=state.repeaterPlan;if(plan){const x=t.px(plan.x),y=t.py(plan.y);g.fillStyle='#f59e0b';g.strokeStyle='#fff';g.lineWidth=2;g.beginPath();g.moveTo(x,y-10);g.lineTo(x+10,y);g.lineTo(x,y+10);g.lineTo(x-10,y);g.closePath();g.fill();g.stroke();g.strokeStyle='#172033';g.lineWidth=2;g.beginPath();g.moveTo(x-4,y);g.lineTo(x+4,y);g.moveTo(x,y-4);g.lineTo(x,y+4);g.stroke();g.fillStyle='#fff';g.font='bold 10px sans-serif';g.fillText('REPEATER',x,y-14);state.markers.push({x,y,type:'router',plan})}const rec=document.getElementById('routerRec');if(rec){if(!state.ml)rec.textContent='Waiting for the RSSI heatmap grid.';else{const before=plan?plan.before:weakCellPercent(state.ml.values);if(!triggered)rec.textContent='Strongest AP: '+(Number.isFinite(strongest)?strongest.toFixed(0)+' dBm':'no APs')+' | Weak cells (< -70 dBm): '+before.toFixed(1)+'%. No repeater suggestion because the strongest AP is not below -70 dBm.';else if(plan)rec.textContent='SUGGESTED REPEATER SPOT (MODEL ESTIMATE) | Strongest AP '+strongest.toFixed(0)+' dBm | Weak cells: '+plan.before.toFixed(1)+'% -> '+plan.after.toFixed(1)+'% | Candidate is at least 2 m from estimated AP markers. Assumes repeater RSSI -50 dBm at 1 m and n='+plan.n.toFixed(2)+'; verify location physically.';else rec.textContent='Strongest AP '+strongest.toFixed(0)+' dBm is below -70 dBm | Weak cells: '+before.toFixed(1)+'% | No grid candidate at least 2 m from estimated AP markers improves weak-cell percentage in this model; collect measured locations or inspect the area.'}}document.getElementById('mapnote').textContent='YOU at centre | '+aps.length+' detected networks | '+t.side.toFixed(1)+' m RSSI-distance square | AP angles schematic | repeater suggestion uses a grid model and is not a measured placement'}\n"
  "function refreshMap(){fetch('/heatmap?layer=all-range').then(r=>{if(!r.ok)throw Error('RSSI heatmap unavailable');return r.json()}).then(h=>{state.ml=h;const aps=state.aps.filter(a=>a.seen),strongest=aps.reduce((m,a)=>Math.max(m,Number(a.r)),-Infinity),world=aps.map(a=>{const d=Number(a.dist)||0,ang=(rangeBearing(a.b)%360)*Math.PI/180;return {x:d*Math.cos(ang),y:d*Math.sin(ang)}});state.repeaterPlan=aps.length&&strongest< -70?planRepeater(h,world):null;draw()}).catch(e=>{state.ml=null;state.repeaterPlan=null;draw();msg(e.message)})}\n"
  "function resetView(){state.view=null;draw()}function zoom(z){const t=transform();state.view={cx:0,cy:0,z:Math.max(.2,Math.min(20,t.v.z*z))};draw()}\n"
  "async function inspectAt(px,py){const t=transform(),x=t.wx(px),y=t.wy(py),h=state.ml;let estimate=null;if(h){const ix=Math.max(0,Math.min(h.nx-1,Math.round((x-h.xmin)/(h.xmax-h.xmin)*(h.nx-1)))),iy=Math.max(0,Math.min(h.ny-1,Math.round((y-h.ymin)/(h.ymax-h.ymin)*(h.ny-1))));estimate=h.values[iy*h.nx+ix]}document.getElementById('inspect').textContent='Schematic position '+x.toFixed(1)+' m east, '+y.toFixed(1)+' m north | predicted strongest RSSI: '+(estimate===null?'no detected Wi-Fi':Number(estimate).toFixed(1)+' dBm')}\n"
  "function canvasPoint(e){const r=cv.getBoundingClientRect();return {x:(e.clientX-r.left)*cv.width/r.width,y:(e.clientY-r.top)*cv.height/r.height}}\n"
  "cv.addEventListener('pointerdown',e=>{cv.setPointerCapture(e.pointerId);const p=canvasPoint(e);state.drag={x:p.x,y:p.y,v:transform().v}});cv.addEventListener('pointermove',e=>{if(!state.drag)return;const p=canvasPoint(e),t=transform(),dx=p.x-state.drag.x,dy=p.y-state.drag.y;state.view={cx:state.drag.v.cx-dx/t.scale,cy:state.drag.v.cy+dy/t.scale,z:state.drag.v.z};draw()});cv.addEventListener('pointerup',e=>{const p=canvasPoint(e);if(state.drag&&Math.hypot(p.x-state.drag.x,p.y-state.drag.y)<5){let hit=null,bd=18*18;for(const m of state.markers){const d=(m.x-p.x)*(m.x-p.x)+(m.y-p.y)*(m.y-p.y);if(d<bd){bd=d;hit=m}}if(hit){if(hit.type==='router'){document.getElementById('inspect').textContent='SUGGESTED REPEATER SPOT (MODEL ESTIMATE) | weak cells before '+hit.plan.before.toFixed(1)+'% -> after '+hit.plan.after.toFixed(1)+'% | at least 2 m from estimated AP markers | virtual repeater assumes -50 dBm at 1 m and n='+hit.plan.n.toFixed(2)+'. Confirm with a real survey.'}else{const a=hit.ap;document.getElementById('inspect').textContent='AP '+hit.index+' | RSSI '+a.r+' dBm | estimated device distance '+Number(a.dist).toFixed(1)+' m | estimated -70 dBm range '+Number(a.useful).toFixed(1)+' m | '+hit.source}}else inspectAt(p.x,p.y)}state.drag=null});cv.addEventListener('wheel',e=>{e.preventDefault();zoom(e.deltaY<0?1.2:.83)},{passive:false});\n"
  "function renderPairs(){const aps=state.aps.filter(a=>a.seen),out=[];for(let i=0;i<aps.length;i++){const a=aps[i];out.push('<span style=\"color:'+apColor(a.b)+'\">AP'+(i+1)+'</span>: '+a.r+' dBm | device range ~'+Number(a.dist).toFixed(1)+' m | modelled -70 dBm range '+Number(a.useful).toFixed(1)+' m')}document.getElementById('networkSummary').innerHTML=out.length?out.join('<br>'):'No Wi-Fi detected in the latest scan.';const pairs=[];for(let i=0;i<aps.length;i++)for(let j=i+1;j<aps.length;j++){const d1=Number(aps[i].dist),d2=Number(aps[j].dist);pairs.push('AP'+(i+1)+' to AP'+(j+1)+': '+Math.abs(d1-d2).toFixed(1)+' to '+(d1+d2).toFixed(1)+' m possible RSSI interval; exact spacing unknown')}document.getElementById('pairwise').textContent=pairs.length?pairs.join(' | '):'Need at least two detected APs.'}\n"
  "\n"
  "async function poll(){try{const r=await fetch('/data'),d=await r.json();state.aps=d.aps||[];renderPairs();document.getElementById('state').textContent=d.surveyPaused?'Paused':d.surveyActive?'Surveying':d.surveyStopped?'Stopped':'Ready';document.getElementById('nets').textContent=state.aps.filter(a=>a.seen).length;document.getElementById('measurements').textContent=d.measurements;document.getElementById('position').textContent='RSSI only';document.getElementById('sd').textContent=d.sd?d.sdRows+' rows':'Unavailable';document.getElementById('session').textContent=d.session;document.getElementById('gpsStatus').textContent='Map uses RSSI only';const key=[d.scans,d.session,state.aps.map(a=>a.b+':'+a.r).join(',')].join('|');if(key!==state.key){state.key=key;state.repeaterPlan=null;refreshMap()}}catch(e){document.getElementById('state').textContent='Connection lost'}}\n"
  "function rangeBearing(b){let h=2166136261;for(let i=0;i<b.length;i++)h=Math.imul(h^b.charCodeAt(i),16777619)>>>0;return h>>>0}\n"
  "poll();setInterval(poll,3000);\n"
  "</script></body></html>\n";

// =====================================================
// MPU6500 REGISTER FUNCTIONS
// =====================================================

bool writeMPURegister(uint8_t reg, uint8_t value) {

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);

  return Wire.endTransmission() == 0;
}

bool readMPURegister(uint8_t reg, uint8_t &value) {

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1) != 1) {
    return false;
  }

  value = Wire.read();
  return true;
}

bool initializeMPU6500() {

  uint8_t chipID = 0;

  if (!readMPURegister(MPU_WHO_AM_I, chipID)) {
    return false;
  }

  Serial.print("MPU WHO_AM_I: 0x");
  Serial.println(chipID, HEX);

  if (chipID != 0x70) {
    return false;
  }

  if (!writeMPURegister(MPU_PWR_MGMT_1, 0x01)) {
    return false;
  }

  delay(100);

  // Accelerometer +/- 8g
  if (!writeMPURegister(MPU_ACCEL_CONFIG, 0x10)) {
    return false;
  }

  // Gyroscope +/- 500 degrees/sec
  if (!writeMPURegister(MPU_GYRO_CONFIG, 0x08)) {
    return false;
  }

  delay(50);

  return true;
}

bool readMPU6500() {

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(MPU_ACCEL_XOUT_H);

  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14) != 14) {
    return false;
  }

  uint8_t data[14];

  for (int i = 0; i < 14; i++) {
    data[i] = Wire.read();
  }

  int16_t rawAX =
    (int16_t)((uint16_t)data[0] << 8 | data[1]);

  int16_t rawAY =
    (int16_t)((uint16_t)data[2] << 8 | data[3]);

  int16_t rawAZ =
    (int16_t)((uint16_t)data[4] << 8 | data[5]);

  int16_t rawGX =
    (int16_t)((uint16_t)data[8] << 8 | data[9]);

  int16_t rawGY =
    (int16_t)((uint16_t)data[10] << 8 | data[11]);

  int16_t rawGZ =
    (int16_t)((uint16_t)data[12] << 8 | data[13]);

  ax = (rawAX / ACCEL_SCALE) * GRAVITY;
  ay = (rawAY / ACCEL_SCALE) * GRAVITY;
  az = (rawAZ / ACCEL_SCALE) * GRAVITY;

  gx = (rawGX / GYRO_SCALE) * (PI / 180.0f);
  gy = (rawGY / GYRO_SCALE) * (PI / 180.0f);
  gz = (rawGZ / GYRO_SCALE) * (PI / 180.0f);

  return true;
}

// =====================================================
// GPS PROCESSING
// =====================================================

void processGPS() {

  while (GPS.available()) {
    gps.encode(GPS.read());
  }
}

bool validGPSFix() {
  return gps.location.isValid() && gps.location.age() < 10000 &&
         (!gps.hdop.isValid() || gps.hdop.hdop() <= 5.0);
}

bool phoneFixValid() {

  return phoneEver &&
         (millis() - phoneFixMs < PHONE_MAX_AGE_MS) &&
         phoneAcc <= PHONE_MAX_ACC_M;
}

// Best available position: 0 = none, 1 = ESP32 GPS module, 2 = phone GPS
int getPosition(double &lat, double &lon) {

  if (validGPSFix()) {
    lat = gps.location.lat();
    lon = gps.location.lng();
    return 1;
  }

  if (phoneFixValid()) {
    lat = phoneLat;
    lon = phoneLon;
    return 2;
  }

  lat = 0;
  lon = 0;
  return 0;
}

const char *posSrcName(int s) {

  return s == 1 ? "module" : (s == 2 ? "phone" : "none");
}

void setGeoOrigin(double lat, double lon) {
  originLat = lat;
  originLon = lon;
  originSet = true;
  logMsg("MODEL", "Local origin set to %.7f, %.7f", lat, lon);
}

// Convert GPS coordinates to a local tangent approximation: X=east, Y=north.
bool gpsToLocal(double lat, double lon, double &x, double &y) {
  if (!originSet) {
    setGeoOrigin(lat, lon);
    gpsAnchorX = manualX;
    gpsAnchorY = manualY;
  }
  const double metersPerDegreeX = 111320.0 * cos(originLat * PI / 180.0);
  const double metersPerDegreeY = 110540.0;
  x = gpsAnchorX + (lon - originLon) * metersPerDegreeX;
  y = gpsAnchorY + (lat - originLat) * metersPerDegreeY;
  // Preserve the last accepted local coordinate if GPS later goes stale.
  manualX = x;
  manualY = y;
  return true;
}

bool getModelPosition(double &x, double &y) {
  if (gpsAuto) {
    double lat, lon;
    if (getPosition(lat, lon)) return gpsToLocal(lat, lon, x, y);
  }
  // Manual coordinates remain valid and scanning never depends on GPS.
  x = manualX;
  y = manualY;
  return true;
}

int findApStats(const char *bssid) {
  for (int i = 0; i < apStatsCount; ++i) {
    if (strcmp(apStats[i].bssid, bssid) == 0) return i;
  }
  return -1;
}

void observeAp(const char *bssid, int rssi) {
  int i = findApStats(bssid);
  if (i < 0) {
    if (apStatsCount >= MAX_AP_STATS) return;
    i = apStatsCount++;
    strncpy(apStats[i].bssid, bssid, sizeof(apStats[i].bssid) - 1);
    apStats[i].bssid[sizeof(apStats[i].bssid) - 1] = '\0';
    apStats[i].sum = 0.0;
    apStats[i].count = 0;
  }
  // Limit the running count to avoid overflow during long sessions.
  if (apStats[i].count >= 60000) {
    apStats[i].sum *= 0.5;
    apStats[i].count /= 2;
  }
  apStats[i].sum += rssi;
  apStats[i].count++;
}

double apMeanFor(const char *bssid) {
  int i = findApStats(bssid);
  if (i >= 0 && apStats[i].count) return apStats[i].sum / apStats[i].count;
  return wifi_ridge::ap_mean_rssi(bssid);
}

float predictAt(const char *bssid, double x, double y) {
  const double apMean = apMeanFor(bssid);
  return (float)(wifi_ridge::predictRSSI(x, y, siteFloorId,
                siteBuildingNum, apMean) + wifi_calibration::RSSI_OFFSET_DBM);
}

double estimatedDeviceRangeM(const char *, double rssiDbm) {
  const double ref=DEFAULT_RSSI_AT_1M_DBM;
  double d=pow(10.0,(ref-rssiDbm)/(10.0*pathLossExponent));
  if(!isfinite(d)) return 10000.0;
  return constrain(d,0.1,10000.0);
}

double usefulRangeM(const char *) {
  const double ref=DEFAULT_RSSI_AT_1M_DBM;
  return constrain(pow(10.0,(ref-USEFUL_SIGNAL_THRESHOLD_DBM)/(10.0*pathLossExponent)),0.1,10000.0);
}

const char *coverageName(int category) {
  switch (category) {
    case 5: return "Excellent";
    case 4: return "Good";
    case 3: return "Fair";
    case 2: return "Poor";
    default: return "Very poor";
  }
}

String getGPSTimestamp() {

  if (!gps.date.isValid() || !gps.time.isValid()) {
    return "";
  }

  char buffer[25];

  snprintf(
    buffer,
    sizeof(buffer),
    "%04d-%02d-%02dT%02d:%02d:%02dZ",
    gps.date.year(),
    gps.date.month(),
    gps.date.day(),
    gps.time.hour(),
    gps.time.minute(),
    gps.time.second()
  );

  return String(buffer);
}

// =====================================================
// CSV FIELD ESCAPING
// =====================================================

String csvEscape(String value) {

  value.replace("\"", "\"\"");

  if (value.indexOf(',') >= 0 ||
      value.indexOf('"') >= 0 ||
      value.indexOf('\n') >= 0) {

    return "\"" + value + "\"";
  }

  return value;
}

// JSON string escaping for the website
String jsonEscape(String value) {

  value.replace("\\", "\\\\");
  value.replace("\"", "\\\"");
  value.replace("\n", " ");
  value.replace("\r", " ");

  return value;
}

// =====================================================
// MICROSD HEADER
// =====================================================

void initializeCSV() {

  if (!sdOK) return;

  if (!SD.exists(filename)) {

    File file = SD.open(filename, FILE_WRITE);

    if (file) {

      file.println("session_id,measurement_id,timestamp,bssid,ssid,rssi,channel,x,y,position_source,gps_lat,gps_lon,gps_accuracy,temperature_C,pressure_hPa");

      file.close();

      Serial.println("Coverage CSV header created.");
      logMsg("SD", "Created %s", filename);
    }
    else {
      Serial.println("CSV header creation failed.");
      logMsg("SD", "Header creation FAILED");
    }
  }
  else {
    Serial.println("Existing coverage CSV found; appending data.");
    logMsg("SD", "Appending to existing %s", filename);
  }
}

// =====================================================
// READ SENSORS
// =====================================================

void readAllSensors() {

  if (mpuOK) {

    if (!readMPU6500()) {
      Serial.println("MPU6500 read error.");
      logMsg("MPU", "Read error");
    }
  }

  if (bmpOK) {

    temperatureC = bmp.readTemperature();
    pressureHpa = bmp.readPressure() / 100.0f;
  }
}

// =====================================================
// SAVE ONE WIFI ACCESS POINT TO CSV
// =====================================================

bool saveNetworkToSD(int networkIndex, unsigned long timeMs,
                     double x, double y) {

  if (!sdOK) return false;

  String ssid = WiFi.SSID(networkIndex);
  String bssid = WiFi.BSSIDstr(networkIndex);

  int rssi = WiFi.RSSI(networkIndex);
  int channel = WiFi.channel(networkIndex);

  double latitude = 0, longitude = 0;
  int posSrc = getPosition(latitude, longitude);   // 0 none, 1 module, 2 phone
  bool fix = posSrc != 0;

  File file = SD.open(filename, FILE_APPEND);

  if (!file) {
    Serial.println("SD: WRITE FAILED");
    logMsg("SD", "Write FAILED for '%s'", ssid.c_str());
    sdFails++;
    return false;
  }

  String timestamp = getGPSTimestamp();
  if (!timestamp.length()) timestamp = String(timeMs);
  const char *source = !gpsAuto ? "manual" :
                       posSrc == 1 ? "gps_module" :
                       posSrc == 2 ? "phone_gps" : "gps_stale";
  file.print(csvEscape(currentSessionId)); file.print(",");
  file.print(scanCounter); file.print(",");
  file.print(csvEscape(timestamp)); file.print(",");
  file.print(bssid); file.print(",");
  file.print(csvEscape(ssid)); file.print(",");
  file.print(rssi); file.print(",");
  file.print(channel); file.print(",");
  file.print(x, 3); file.print(",");
  file.print(y, 3); file.print(",");
  file.print(source); file.print(",");
  if (fix) file.print(latitude, 7); file.print(",");
  if (fix) file.print(longitude, 7); file.print(",");
  if (gpsAuto && posSrc == 2) file.print(phoneAcc, 1);
  file.print(",");
  file.print(temperatureC, 2); file.print(",");
  file.println(pressureHpa, 2);

  file.close();

  sdRows++;

  Serial.print("SAVED: ");
  Serial.print(ssid);
  Serial.print(" | RSSI: ");
  Serial.print(rssi);
  Serial.println(" dBm");

  return true;
}

// =====================================================
// MAP SAMPLES (for the website heatmap)
// =====================================================

void addSample(double x, double y, int rssi, float predicted, const char *bssid,
               const char *ssid, int channel, bool gpsValid, uint32_t timestamp) {
  Sample &s = samples[sampleHead];
  s.measurementID = scanCounter;
  s.timestamp = timestamp;
  s.x = (float)x;
  s.y = (float)y;
  if(!surveyHasBounds){surveyMinX=surveyMaxX=x;surveyMinY=surveyMaxY=y;surveyHasBounds=true;}
  else {surveyMinX=min(surveyMinX,x);surveyMaxX=max(surveyMaxX,x);surveyMinY=min(surveyMinY,y);surveyMaxY=max(surveyMaxY,y);}
  s.rssi = (int8_t)rssi;
  s.channel = (uint8_t)channel;
  s.predicted = predicted;
  s.gpsValid = gpsValid;
  strncpy(s.ssid, ssid, sizeof(s.ssid) - 1);
  s.ssid[sizeof(s.ssid) - 1] = '\0';
  strncpy(s.bssid, bssid, sizeof(s.bssid) - 1);
  s.bssid[sizeof(s.bssid) - 1] = '\0';

  sampleHead = (sampleHead + 1) % MAX_SAMPLES;

  if (sampleCount < MAX_SAMPLES) sampleCount++;
}

uint32_t rangeMapHash(const char *bssid) {
  uint32_t hash=2166136261UL;
  for(int i=0;bssid[i];i++){hash^=(uint8_t)bssid[i];hash*=16777619UL;}
  return hash;
}

void startNewSurvey() {
  currentSessionId = String((uint32_t)ESP.getEfuseMac(), HEX) + "-" +
                     String((uint32_t)esp_random(), HEX);
  surveyStarted = true;
  surveyPaused = false;
  surveyStopped = false;
  surveyMeasurementCount = 0;
  surveyHasBounds = false;
  surveyMinX=surveyMaxX=surveyMinY=surveyMaxY=0.0;
  manualX = 0.0;
  manualY = 0.0;
  gpsAuto = false;
  originSet = false;
  gpsAnchorX = gpsAnchorY = 0.0;
  sampleHead = sampleCount = 0;
  liveApCount = 0;
  nextScanTime = millis();
  logMsg("SURVEY", "New session %s started at local origin (0,0)", currentSessionId.c_str());
}

// =====================================================
// PROCESS WIFI SCAN RESULTS
// =====================================================

void processScanResults(int count) {

  networksFound = count;
  strongestRSSI = -127;
  strongestSSID = "";

  Serial.println();
  Serial.println("======= WIFI SCAN RESULTS =======");

  Serial.print("Networks found: ");
  Serial.println(count);

  unsigned long scanTime = millis();

  readAllSensors();

  double modelX = 0.0, modelY = 0.0;
  const bool modelPositionValid = getModelPosition(modelX, modelY);

  int targetRssi = -127;
  bool targetFound = false;
  int rowsSaved = 0;

  for (int i = 0; i < count; i++) {

    int rssi = WiFi.RSSI(i);

    Serial.println("-----------------------------");

    Serial.print("SSID: ");
    Serial.println(WiFi.SSID(i));

    Serial.print("BSSID: ");
    Serial.println(WiFi.BSSIDstr(i));

    Serial.print("RSSI: ");
    Serial.print(rssi);
    Serial.println(" dBm");

    Serial.print("Channel: ");
    Serial.println(WiFi.channel(i));

    Serial.print("Encryption: ");
    Serial.println(
      WiFi.encryptionType(i) == WIFI_AUTH_OPEN
      ? "OPEN"
      : "SECURED"
    );

    if (rssi > strongestRSSI) {
      strongestRSSI = rssi;
      strongestSSID = WiFi.SSID(i);
    }

    String bssid = WiFi.BSSIDstr(i);
    observeAp(bssid.c_str(), rssi);
    const double apMean = apMeanFor(bssid.c_str());
    const bool predictionValid = WIFI_RIDGE_MODEL_AVAILABLE && modelPositionValid;
    const float predicted = predictionValid ? predictAt(bssid.c_str(), modelX, modelY) : NAN;

    // Keep live measurements and on-device model output for the website.
    int liveIndex=0;
    while(liveIndex<liveApCount && strcmp(liveAps[liveIndex].bssid,bssid.c_str())!=0)liveIndex++;
    if(liveIndex<MAX_LIVE_APS){
      if(liveIndex==liveApCount)liveApCount++;
      LiveAp &a = liveAps[liveIndex];
      WiFi.SSID(i).toCharArray(a.ssid, sizeof(a.ssid));
      bssid.toCharArray(a.bssid, sizeof(a.bssid));
      a.seenScan=scanCounter;
      a.rssi = rssi;
      a.ch = WiFi.channel(i);
      a.open = (WiFi.encryptionType(i) == WIFI_AUTH_OPEN);
      a.apMean = apMean;
      a.predicted = predicted;
      a.coverage = predictionValid ? wifi_ridge::coverageClass(predicted) : 0;
      a.predictionValid = predictionValid;
      a.distanceM=estimatedDeviceRangeM(bssid.c_str(),rssi);
      a.usefulRangeM=usefulRangeM(bssid.c_str());
    }

    if (TARGET_SSID[0] != 0 && WiFi.SSID(i) == TARGET_SSID && rssi > targetRssi) {
      targetRssi = rssi;
      targetFound = true;
    }

    if (surveyStarted && !surveyPaused && !surveyStopped) {
      if (saveNetworkToSD(i, scanTime, modelX, modelY)) rowsSaved++;
      double sampleLat=0,sampleLon=0;
      const int gpsSource=getPosition(sampleLat,sampleLon);
      addSample(modelX,modelY,rssi,predicted,bssid.c_str(),WiFi.SSID(i).c_str(),
                WiFi.channel(i),gpsSource!=0,(uint32_t)scanTime);
      surveyMeasurementCount++;
    }
  }

  Serial.println("-----------------------------");

  Serial.print("GPS fix: ");
  Serial.println(validGPSFix() ? "VALID" : "NOT AVAILABLE");

  if (validGPSFix()) {
    Serial.print("Latitude: ");
    Serial.println(gps.location.lat(), 6);

    Serial.print("Longitude: ");
    Serial.println(gps.location.lng(), 6);
  }

  Serial.print("Strongest network: ");
  Serial.println(strongestSSID);

  Serial.print("Strongest RSSI: ");
  Serial.println(strongestRSSI);

  Serial.println("================================");

  // ---- structured log + map sample ----
  logMsg("SCAN", "#%lu done in %lu ms: %d networks, strongest '%s' %d dBm",
         scanCounter, millis() - scanStartMs, count,
         strongestSSID.c_str(), strongestRSSI);

  if (sdOK) {
    logMsg("SD", "Saved %d/%d rows (total %lu, failures %lu)",
           rowsSaved, count, sdRows, sdFails);
  } else {
    logMsg("SD", "Not available - %d rows NOT saved", count);
  }

  if (modelPositionValid) {
    logMsg("MODEL", "Inference at local X=%.2f Y=%.2f floor=%d building=%d; %d readings stored",
           modelX, modelY, siteFloorId, siteBuildingNum, sampleCount);
  } else {
    logMsg("MODEL", "No coordinate available. Enter X/Y on the dashboard to enable spatial inference.");
  }

  if (TARGET_SSID[0] != 0 && !targetFound) {
    logMsg("SCAN", "Target '%s' NOT seen in this scan", TARGET_SSID);
  }

  WiFi.scanDelete();
}

// =====================================================
// START AND CHECK ASYNCHRONOUS WIFI SCAN
// =====================================================

void startWiFiScan() {

  if (scanInProgress) return;

  Serial.println();
  Serial.println("Starting Wi-Fi scan...");

  scanCounter++;
  scanStartMs = millis();
  logMsg("SCAN", "#%lu starting", scanCounter);

  WiFi.scanDelete();

  WiFi.scanNetworks(true, true);

  scanInProgress = true;
}

void checkWiFiScan() {

  if (!scanInProgress) return;

  int result = WiFi.scanComplete();

  if (result == WIFI_SCAN_RUNNING) {
    return;
  }

  scanInProgress = false;

  if (result == WIFI_SCAN_FAILED) {

    Serial.println("Wi-Fi scan failed.");
    logMsg("SCAN", "#%lu FAILED after %lu ms", scanCounter, millis() - scanStartMs);

    WiFi.scanDelete();
    nextScanTime = millis() + scanInterval;
    return;
  }

  processScanResults(result);

  nextScanTime = millis() + scanInterval;
}

// =====================================================
// WEB HANDLERS
// =====================================================

void handleRoot() {

  logMsg("HTTP", "GET / from %s", server.client().remoteIP().toString().c_str());
  server.send_P(200, "text/html", PAGE);
}

void handleData() {

#if LOG_HTTP_POLLS
  logMsg("HTTP", "GET /data from %s", server.client().remoteIP().toString().c_str());
#endif

  bool fix = validGPSFix();

  String j;
  j.reserve(7000);

  j += "{\"up\":" + String(millis());
  double pLat, pLon;
  int posSrc = getPosition(pLat, pLon);
  double modelX = 0.0, modelY = 0.0;
  const bool modelReady = getModelPosition(modelX, modelY);
  j += ",\"fix\":" + String(fix ? 1 : 0);
  j += ",\"pos\":" + String(posSrc);
  j += ",\"pacc\":" + String((int)phoneAcc);
  j += ",\"sats\":" + String((unsigned)gps.satellites.value());
  j += ",\"temp\":" + String(temperatureC, 1);
  j += ",\"press\":" + String(pressureHpa, 1);
  j += ",\"scans\":" + String(scanCounter);
  j += ",\"nets\":" + String(networksFound);
  j += ",\"sd\":" + String(sdOK ? 1 : 0);
  j += ",\"sdRows\":" + String(sdRows);
  j += ",\"samples\":" + String(sampleCount);
  j += ",\"measurements\":" + String(surveyMeasurementCount);
  j += ",\"modelAvailable\":" + String(WIFI_RIDGE_MODEL_AVAILABLE ? 1 : 0);
  j += ",\"modelReady\":" + String(modelReady ? 1 : 0);
  j += ",\"manual\":" + String(gpsAuto ? 0 : 1);
  j += ",\"gpsAuto\":" + String(gpsAuto ? 1 : 0);
  j += ",\"surveyActive\":" + String(surveyStarted && !surveyStopped && !surveyPaused ? 1 : 0);
  j += ",\"surveyPaused\":" + String(surveyPaused ? 1 : 0);
  j += ",\"surveyStopped\":" + String(surveyStopped ? 1 : 0);
  j += ",\"session\":\"" + jsonEscape(currentSessionId) + "\"";
  j += ",\"x\":" + (modelReady ? String(modelX, 2) : String("null"));
  j += ",\"y\":" + (modelReady ? String(modelY, 2) : String("null"));
  j += ",\"hasBounds\":" + String(surveyHasBounds ? 1 : 0);
  j += ",\"minX\":" + (surveyHasBounds?String(surveyMinX,2):String("null"));
  j += ",\"maxX\":" + (surveyHasBounds?String(surveyMaxX,2):String("null"));
  j += ",\"minY\":" + (surveyHasBounds?String(surveyMinY,2):String("null"));
  j += ",\"maxY\":" + (surveyHasBounds?String(surveyMaxY,2):String("null"));
  j += ",\"floor\":" + String(siteFloorId);
  j += ",\"building\":" + String(siteBuildingNum);
  j += ",\"originSet\":" + String(originSet ? 1 : 0);
  j += ",\"originLat\":" + (originSet ? String(originLat, 7) : String("null"));
  j += ",\"originLon\":" + (originSet ? String(originLon, 7) : String("null"));
  j += ",\"modelOffset\":" + String(wifi_calibration::RSSI_OFFSET_DBM, 2);
  j += ",\"target\":\"" + jsonEscape(String(TARGET_SSID)) + "\"";
  j += ",\"aps\":[";

  for (int i = 0; i < liveApCount; i++) {

    if (i) j += ",";

    j += "{\"s\":\"" + jsonEscape(String(liveAps[i].ssid)) + "\"";
    j += ",\"b\":\"" + jsonEscape(String(liveAps[i].bssid)) + "\"";
    j += ",\"r\":" + String(liveAps[i].rssi);
    j += ",\"seen\":"+String(liveAps[i].seenScan==scanCounter?1:0);
    j += ",\"c\":" + String(liveAps[i].ch);
    j += ",\"o\":" + String(liveAps[i].open ? 1 : 0);
    j += ",\"mean\":" + String(liveAps[i].apMean, 1);
    j += ",\"pred\":" + (liveAps[i].predictionValid ? String(liveAps[i].predicted, 1) : String("null"));
    j += ",\"cls\":\"" + String(liveAps[i].predictionValid ? coverageName(liveAps[i].coverage) : "") + "\"";
    j += ",\"dist\":"+String(liveAps[i].distanceM,2)+",\"useful\":"+String(liveAps[i].usefulRangeM,2)+"}";
  }

  j += "]}";

  server.send(200, "application/json", j);
}

void handlePoints() {
  const String requested = server.hasArg("bssid") ? server.arg("bssid") : "";
  server.setContentLength(CONTENT_LENGTH_UNKNOWN);
  server.send(200, "application/json", "");
  server.sendContent("[");
  String out;
  out.reserve(5200);
  int emitted = 0;
  for (int i = 0; i < sampleCount; ++i) {
    const int idx = (sampleHead - sampleCount + i + MAX_SAMPLES) % MAX_SAMPLES;
    const Sample &s = samples[idx];
    if (requested.length() && requested != s.bssid) continue;
    if (emitted++) out += ",";
    out += "[" + String(s.x, 3) + "," + String(s.y, 3) + "," +
           String((int)s.rssi) + ",";
    out += isfinite(s.predicted) ? String(s.predicted, 2) : String("null");
    out += "," + String(s.measurementID) + "," + String(s.timestamp) + ",\"" +
           jsonEscape(String(s.bssid)) + "\",\"" + jsonEscape(String(s.ssid)) +
           "\"," + String(s.channel) + "," + String(s.gpsValid ? 1 : 0) + "]";
    if (out.length() > 5000) {
      server.sendContent(out);
      out = "";
    }
  }
  out += "]";
  server.sendContent(out);
  server.sendContent("");
}

void handleHeatmap() {
  const String requestedLayer=server.hasArg("layer")?server.arg("layer"):"all-range";
  if(requestedLayer!="all-range"){
    server.send(400,"application/json","{\"error\":\"this map uses the RSSI-only distance model\"}");return;
  }
  double maxDistance=0.0;
  for(int i=0;i<liveApCount;i++)if(liveAps[i].seenScan==scanCounter)
    maxDistance=max(maxDistance,liveAps[i].distanceM);
  const double radius=max(10.0,maxDistance+8.0);
  const double xmin=-radius,xmax=radius,ymin=-radius,ymax=radius;
  const int NX=32,NY=32;
  String j;j.reserve(9000);
  j="{\"layer\":\"all-range\",\"basis\":\"RSSI distance only; directions schematic\",\"nx\":"+String(NX)+
    ",\"ny\":"+String(NY)+",\"pathLossExponent\":"+String(pathLossExponent,2)+",\"xmin\":"+String(xmin,2)+",\"xmax\":"+String(xmax,2)+
    ",\"ymin\":"+String(ymin,2)+",\"ymax\":"+String(ymax,2)+",\"values\":[";
  for(int iy=0;iy<NY;iy++){
    const double y=ymin+(ymax-ymin)*iy/(NY-1);
    for(int ix=0;ix<NX;ix++){
      const double x=xmin+(xmax-xmin)*ix/(NX-1);
      if(iy||ix)j+=",";
      double strongest=-1000.0;bool hasValue=false;
      for(int ap=0;ap<liveApCount;ap++){
        const LiveAp &a=liveAps[ap];if(a.seenScan!=scanCounter)continue;
        const double range=max(0.1,estimatedDeviceRangeM(a.bssid,a.rssi));
        const double angle=(rangeMapHash(a.bssid)%360)*PI/180.0;
        const double apX=range*cos(angle),apY=range*sin(angle);
        const double fromAp=max(0.1,hypot(x-apX,y-apY));
        const double predicted=constrain((double)a.rssi-
          10.0*pathLossExponent*log10(fromAp/range),-100.0,-30.0);
        if(!hasValue||predicted>strongest){strongest=predicted;hasValue=true;}
      }
      if(hasValue)j+=String(strongest,1);else j+="null";
    }
    yield();
  }
  j+="]}";
  server.send(200,"application/json",j);
}

void handleAps() {
  String j="[";
  for(int i=0;i<liveApCount;i++){
    if(i)j+=",";
    j+="{\"s\":\""+jsonEscape(String(liveAps[i].ssid))+"\",\"b\":\""+
       jsonEscape(String(liveAps[i].bssid))+"\",\"r\":"+String(liveAps[i].rssi)+
       ",\"c\":"+String(liveAps[i].ch)+",\"pred\":"+
       (liveAps[i].predictionValid?String(liveAps[i].predicted,1):String("null"))+
       ",\"dist\":"+String(liveAps[i].distanceM,2)+
       ",\"useful\":"+String(liveAps[i].usefulRangeM,2)+"}";
  }
  j+="]";
  server.send(200,"application/json",j);
}

void surveyReply(const char *message) {
  String j="{\"ok\":1,\"message\":\""+jsonEscape(String(message))+"\",\"session\":\""+
           jsonEscape(currentSessionId)+"\"}";
  server.send(200,"application/json",j);
}
void handleSurveyStart() { startNewSurvey(); surveyReply("survey started at (0,0)"); }
void handleSurveyPause() {
  if(!surveyStarted || surveyStopped){server.send(409,"text/plain","no active survey");return;}
  surveyPaused=true; surveyReply("survey paused; scans continue without recording");
}
void handleSurveyResume() {
  if(!surveyStarted || surveyStopped) startNewSurvey();
  surveyPaused=false; surveyStopped=false; nextScanTime=millis(); surveyReply("survey resumed");
}
void handleSurveyStop() {
  if(!surveyStarted){server.send(409,"text/plain","no survey to stop");return;}
  surveyPaused=false; surveyStopped=true; surveyReply("survey stopped; SD history retained");
}
void handleSurveyClear() { startNewSurvey(); surveyReply("new survey started; previous SD history retained"); }

void handleSurveyHistory() {
  if(!sdOK){server.send(503,"application/json","{\"error\":\"SD unavailable\"}");return;}
  File file=SD.open(filename,FILE_READ);
  if(!file){server.send(404,"application/json","{\"error\":\"survey log missing\"}");return;}
  char line[384];
  file.readBytesUntil('\n',line,sizeof(line)); // CSV header
  String ids[20]; uint32_t counts[20]={0}; int used=0;
  while(file.available()){
    size_t n=file.readBytesUntil('\n',line,sizeof(line)-1);
    if(!n)continue; line[n]='\0';
    char *comma=strchr(line,','); if(!comma)continue; *comma='\0';
    int i=0; while(i<used && ids[i]!=line)i++;
    if(i==used){if(used==20){for(int k=1;k<20;k++){ids[k-1]=ids[k];counts[k-1]=counts[k];}used=19;}ids[used]=line;counts[used]=0;used++;}
    counts[i]++;
  }
  file.close();
  String j="[";
  for(int i=0;i<used;i++){if(i)j+=",";j+="{\"session\":\""+jsonEscape(ids[i])+"\",\"measurements\":"+String(counts[i])+"}";}
  j+="]";server.send(200,"application/json",j);
}

void handlePredict() {
#if !WIFI_RIDGE_MODEL_AVAILABLE
  server.send(503,"application/json","{\"error\":\"no embedded ML model\"}");
  return;
#else
  double x=0,y=0;
  if (!getModelPosition(x,y)) {
    server.send(409,"application/json","{\"error\":\"set local X/Y or wait for GPS\"}");
    return;
  }
  String j="{\"x\":"+String(x,2)+",\"y\":"+String(y,2)+",\"floor\":"+
           String(siteFloorId)+",\"building\":"+String(siteBuildingNum)+",\"aps\":[";
  for(int i=0;i<liveApCount;i++){
    if(i)j+=",";
    const float pred=predictAt(liveAps[i].bssid,x,y);
    const int cls=wifi_ridge::coverageClass(pred);
    j+="{\"bssid\":\""+jsonEscape(String(liveAps[i].bssid))+"\",\"rssi\":"+
       String(pred,2)+",\"class\":\""+String(coverageName(cls))+"\"}";
  }
  j+="]}";
  server.send(200,"application/json",j);
#endif
}

void handlePosition() {
  if (server.hasArg("mode") && server.arg("mode") == "gps") {
    double oldX=manualX,oldY=manualY;
    getModelPosition(oldX,oldY);
    manualX=oldX;manualY=oldY;gpsAnchorX=oldX;gpsAnchorY=oldY;
    originSet=false;
    gpsAuto=true;
    double lat,lon,x,y;
    if(getPosition(lat,lon))gpsToLocal(lat,lon,x,y);
    server.send(200,"text/plain","GPS Auto enabled; current coordinates retained until a fix is available");
    return;
  }
  if(server.hasArg("dx")&&server.hasArg("dy")){
    double x=manualX,y=manualY;getModelPosition(x,y);
    const double dx=server.arg("dx").toDouble(),dy=server.arg("dy").toDouble();
    if(!isfinite(dx)||!isfinite(dy)||fabs(dx)>100||fabs(dy)>100){server.send(400,"text/plain","invalid movement step");return;}
    manualX=x+dx;manualY=y+dy;gpsAuto=false;
    server.send(200,"text/plain","position moved; manual relative mode enabled");return;
  }
  if (!server.hasArg("x") || !server.hasArg("y")) {
    server.send(400,"text/plain","provide x and y in metres");
    return;
  }
  const double nextX=server.arg("x").toDouble(), nextY=server.arg("y").toDouble();
  if(!isfinite(nextX)||!isfinite(nextY)||fabs(nextX)>10000||fabs(nextY)>10000){
    server.send(400,"text/plain","X/Y must be finite metres within +/-10000"); return;
  }
  const int nextFloor=server.hasArg("f")?server.arg("f").toInt():siteFloorId;
  const int nextBuilding=server.hasArg("b")?server.arg("b").toInt():siteBuildingNum;
  if(nextFloor < -20 || nextFloor > 100 || nextBuilding < 0 || nextBuilding > 100){
    server.send(400,"text/plain","invalid floor or building index"); return;
  }
  manualX=nextX; manualY=nextY;
  siteFloorId=nextFloor; siteBuildingNum=nextBuilding;
  gpsAuto=false;
  server.send(200,"text/plain","local position updated");
}

void handleOrigin() {
  if(!server.hasArg("lat")||!server.hasArg("lon")){
    server.send(400,"text/plain","provide origin lat and lon"); return;
  }
  const double lat=server.arg("lat").toDouble(), lon=server.arg("lon").toDouble();
  if(lat < -90 || lat > 90 || lon < -180 || lon > 180 || (lat==0 && lon==0)){
    server.send(400,"text/plain","invalid origin"); return;
  }
  double x=manualX,y=manualY;getModelPosition(x,y);
  manualX=x;manualY=y;gpsAnchorX=x;gpsAnchorY=y;
  setGeoOrigin(lat,lon);gpsAuto=true;
  server.send(200,"text/plain","GPS origin updated; X east, Y north");
}

void handleCSV() {

  logMsg("HTTP", "GET /csv from %s", server.client().remoteIP().toString().c_str());

  if (!sdOK) {
    server.send(503, "text/plain", "SD card not ready");
    return;
  }

  File file = SD.open(filename, FILE_READ);

  if (!file) {
    logMsg("HTTP", "CSV open failed");
    server.send(404, "text/plain", "CSV not found");
    return;
  }

  const String selectedSession=server.hasArg("session")?server.arg("session"):"";
  server.sendHeader("Content-Disposition", selectedSession.length()?
                    "attachment; filename=coverage_ai_session.csv":
                    "attachment; filename=coverage_ai_survey.csv");
  size_t sent=0;
  if(!selectedSession.length()) sent=server.streamFile(file,"text/csv");
  else {
    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200,"text/csv","");
    String chunk;chunk.reserve(1024);
    char line[384];bool first=true;
    while(file.available()){
      size_t n=file.readBytesUntil('\n',line,sizeof(line)-1);
      if(!n)continue;line[n]='\0';
      bool keep=first;
      if(first)first=false;
      else {char *comma=strchr(line,',');if(comma){*comma='\0';keep=selectedSession==line;*comma=',';}}
      if(keep){chunk+=line;chunk+='\n';sent+=n+1;}
      if(chunk.length()>=900){server.sendContent(chunk);chunk="";}
    }
    if(chunk.length())server.sendContent(chunk);
    server.sendContent("");
  }
  file.close();

  logMsg("HTTP", "CSV sent: %u bytes", (unsigned)sent);
}

// Phone shares its GPS position (used when the GPS module has no fix)
void handlePhoneLocation() {

  if (!server.hasArg("lat") || !server.hasArg("lon")) {
    server.send(400, "text/plain", "need lat and lon");
    return;
  }

  double lat = server.arg("lat").toDouble();
  double lon = server.arg("lon").toDouble();
  float acc = server.hasArg("acc") ? server.arg("acc").toFloat() : 10.0f;

  if (lat < -90 || lat > 90 || lon < -180 || lon > 180 || (lat == 0 && lon == 0)) {
    logMsg("PHONE", "Rejected invalid position lat=%s lon=%s",
           server.arg("lat").c_str(), server.arg("lon").c_str());
    server.send(400, "text/plain", "invalid position");
    return;
  }

  phoneLat = lat;
  phoneLon = lon;
  phoneAcc = (acc > 0) ? acc : 10.0f;
  phoneFixMs = millis();

  static unsigned long lastLog = 0;

  if (!phoneEver) {
    phoneEver = true;
    lastLog = 0;
    logMsg("PHONE", "First phone GPS received from %s",
           server.client().remoteIP().toString().c_str());
  }

  if (millis() - lastLog > 10000 || lastLog == 0) {
    lastLog = millis();
    logMsg("PHONE", "GPS %.6f, %.6f accuracy %.0f m (%s)", lat, lon, phoneAcc,
           phoneAcc <= PHONE_MAX_ACC_M ? "usable" : "too inaccurate - ignored");
  }

  server.send(200, "text/plain", "ok");
}

// Send the phone to the dashboard (this is what triggers the pop-up)
void redirectToPortal() {

  String url = "http://" + WiFi.softAPIP().toString() + "/";

  server.sendHeader("Location", url, true);
  server.sendHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  server.send(302, "text/plain", "");
}

// Connectivity-check URLs used by Android / iOS / Windows / Firefox.
// A reply that is not their expected "internet OK" answer = "sign-in needed" pop-up.
void handlePortalProbe() {

  logMsg("PORTAL", "Probe %s%s from %s -> redirect to dashboard",
         server.hostHeader().c_str(), server.uri().c_str(),
         server.client().remoteIP().toString().c_str());

  redirectToPortal();
}

// Any other URL (any website the phone tries) also lands on the dashboard
void handleNotFound() {

  logMsg("PORTAL", "%s%s -> redirect to dashboard",
         server.hostHeader().c_str(), server.uri().c_str());

  redirectToPortal();
}

// =====================================================
// OLED DASHBOARD
// =====================================================

void updateOLED() {

  if (!oledOK) return;

  display.clearDisplay();

  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  display.setCursor(0, 0);
  display.println("COVERAGE-AI LIVE");

  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  display.setCursor(0, 13);

  display.print("GPS: ");
  display.print(validGPSFix() ? "FIX" : (phoneFixValid() ? "PHN" : "WAIT"));

  display.print(" SAT:");
  display.println(gps.satellites.value());

  display.setCursor(0, 23);

  display.print("WIFI APs: ");
  display.println(networksFound);

  display.setCursor(0, 33);

  display.print("RSSI: ");

  if (networksFound > 0) {
    display.print(strongestRSSI);
    display.println(" dBm");
  }
  else {
    display.println("--");
  }

  display.setCursor(0, 43);
  display.print("ML: ");
  if (liveApCount > 0 && liveAps[0].predictionValid) {
    display.print(liveAps[0].predicted, 0);
    display.print(" dBm ");
    display.println(coverageName(liveAps[0].coverage));
  } else {
    display.println(WIFI_RIDGE_MODEL_AVAILABLE ? "waiting for AP" : "ML not loaded");
  }

  display.setCursor(0, 53);

  display.print("T:"); display.print(temperatureC, 1); display.print("C P:");
  display.print(pressureHpa, 0); display.println("hPa");

  display.display();
}

// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("==================================");
  Serial.println("       COVERAGE-AI SYSTEM");
  Serial.println("==================================");

  logMsg("BOOT", "Free heap: %u bytes", (unsigned)ESP.getFreeHeap());

  logMsg("RANGE", "RSSI range model uses RSSI@1m=%.1f dBm, n=%.2f",DEFAULT_RSSI_AT_1M_DBM,pathLossExponent);

  // I2C
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);

  // OLED
  oledOK = display.begin(
    SSD1306_SWITCHCAPVCC,
    OLED_ADDRESS
  );

  if (oledOK) {

    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 10);

    display.println("COVERAGE-AI");
    display.println("Initializing...");

    display.display();

    Serial.println("OLED: ONLINE");
  }
  else {
    Serial.println("OLED: ERROR");
  }
  logMsg("BOOT", "OLED %s", oledOK ? "online" : "ERROR");

  // MPU6500
  mpuOK = initializeMPU6500();

  Serial.println(
    mpuOK ? "MPU6500: ONLINE" : "MPU6500: ERROR"
  );
  logMsg("BOOT", "MPU6500 %s", mpuOK ? "online" : "ERROR");

  // BMP280
  bmpOK = bmp.begin(0x76);

  if (bmpOK) {

    bmp.setSampling(
      Adafruit_BMP280::MODE_NORMAL,
      Adafruit_BMP280::SAMPLING_X2,
      Adafruit_BMP280::SAMPLING_X16,
      Adafruit_BMP280::FILTER_X16,
      Adafruit_BMP280::STANDBY_MS_500
    );
  }

  Serial.println(
    bmpOK ? "BMP280: ONLINE" : "BMP280: ERROR"
  );
  logMsg("BOOT", "BMP280 %s", bmpOK ? "online" : "ERROR");

  // GPS
  GPS.begin(9600, SERIAL_8N1, GPS_RX, -1);
  gpsOK = true;

  Serial.println("GPS UART: ONLINE");
  logMsg("BOOT", "GPS UART started (9600 baud, RX=%d); waiting for fix", GPS_RX);

  // MicroSD
  sdSPI.begin(
    SD_SCK,
    SD_MISO,
    SD_MOSI,
    SD_CS
  );

  sdOK = SD.begin(
    SD_CS,
    sdSPI,
    4000000
  );

  Serial.println(
    sdOK ? "MICROSD: ONLINE" : "MICROSD: ERROR"
  );

  if (sdOK) {
    logMsg("BOOT", "SD online, card size %llu MB", SD.cardSize() / (1024ULL * 1024ULL));
  } else {
    logMsg("BOOT", "SD ERROR - data will NOT be saved");
  }

  initializeCSV();

  // Wi-Fi: scanner (STA) + own network for the website (AP)
  WiFi.onEvent(onWiFiEvent);
  WiFi.mode(WIFI_AP_STA);

  bool apOK = WiFi.softAP(AP_SSID, AP_PASS);
  WiFi.disconnect();

  delay(500);

  logMsg("WIFI", "softAP '%s' %s, IP %s", AP_SSID,
         apOK ? "started" : "FAILED", WiFi.softAPIP().toString().c_str());

  // Website
  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.on("/aps", handleAps);
  server.on("/points", handlePoints);
  server.on("/heatmap", handleHeatmap);
  server.on("/predict", handlePredict);
  server.on("/csv", handleCSV);
  server.on("/survey/history", handleSurveyHistory);
  server.on("/loc", handlePhoneLocation);
  server.on("/position", handlePosition);
  server.on("/origin", handleOrigin);
  server.on("/survey/start", handleSurveyStart);
  server.on("/survey/pause", handleSurveyPause);
  server.on("/survey/resume", handleSurveyResume);
  server.on("/survey/stop", handleSurveyStop);
  server.on("/survey/clear", handleSurveyClear);

  // captive-portal probe URLs
  server.on("/generate_204", handlePortalProbe);         // Android
  server.on("/gen_204", handlePortalProbe);              // Android
  server.on("/hotspot-detect.html", handlePortalProbe);  // iOS / macOS
  server.on("/library/test/success.html", handlePortalProbe);  // iOS / macOS
  server.on("/connecttest.txt", handlePortalProbe);      // Windows
  server.on("/ncsi.txt", handlePortalProbe);             // Windows
  server.on("/redirect", handlePortalProbe);             // Windows
  server.on("/canonical.html", handlePortalProbe);       // Firefox
  server.on("/success.txt", handlePortalProbe);          // Firefox
  server.onNotFound(handleNotFound);
  server.begin();

  logMsg("WEB", "Server started");

  dnsServer.start(DNS_PORT, "*", WiFi.softAPIP());
  logMsg("PORTAL", "Captive portal DNS started (all domains -> %s)",
         WiFi.softAPIP().toString().c_str());

  Serial.println("WIFI SCANNER: READY");

  Serial.println("----------------------------------");
  Serial.println("COVERAGE-AI INITIALIZATION COMPLETE");
  Serial.println("----------------------------------");

  logMsg("BOOT", "Join Wi-Fi '%s' (password '%s') and open http://%s",
         AP_SSID, AP_PASS, WiFi.softAPIP().toString().c_str());
  logMsg("BOOT", "Mapping target: %s", TARGET_SSID[0] ? TARGET_SSID : "(strongest AP)");

  nextScanTime = millis() + 1000;
}

// =====================================================
// MAIN LOOP
// =====================================================

void loop() {

  // Captive portal DNS + website
  dnsServer.processNextRequest();
  server.handleClient();

  // Continuously process GPS data
  processGPS();

  unsigned long now = millis();

  // Log GPS fix changes
  bool fixNow = validGPSFix();

  if (fixNow != lastFixState) {

    lastFixState = fixNow;

    if (fixNow) {
      logMsg("GPS", "FIX acquired: %.6f, %.6f (%u satellites)",
             gps.location.lat(), gps.location.lng(),
             (unsigned)gps.satellites.value());
    } else {
      logMsg("GPS", "FIX lost");
    }
  }

  // Log position source changes (module / phone / none)
  double tmpLat, tmpLon;
  int srcNow = getPosition(tmpLat, tmpLon);

  if (srcNow != lastPosSrc) {
    logMsg("POS", "Position source: %s -> %s", posSrcName(lastPosSrc), posSrcName(srcNow));
    lastPosSrc = srcNow;
  }

  // Read sensors once per second
  if (now - lastSensorRead >= sensorInterval) {

    lastSensorRead = now;

    readAllSensors();

    Serial.print("GPS: ");
    Serial.print(validGPSFix() ? "FIX" : "WAIT");

    if (validGPSFix()) {
      Serial.print(" | LAT: ");
      Serial.print(gps.location.lat(), 6);

      Serial.print(" | LON: ");
      Serial.print(gps.location.lng(), 6);
    }

    Serial.print(" | TEMP: ");
    Serial.print(temperatureC, 1);

    Serial.print(" C | PRESSURE: ");
    Serial.print(pressureHpa, 1);

    Serial.println(" hPa");
  }

  // Update OLED
  if (now - lastOLEDUpdate >= oledInterval) {

    lastOLEDUpdate = now;

    updateOLED();
  }

  // Heartbeat log
  if (now - lastHeartbeat >= heartbeatInterval) {

    lastHeartbeat = now;

    logMsg("BEAT", "heap=%u clients=%d scans=%lu samples=%d sdRows=%lu gps=%s sats=%u",
           (unsigned)ESP.getFreeHeap(), WiFi.softAPgetStationNum(),
           scanCounter, sampleCount, sdRows,
           posSrcName(lastPosSrc),
           (unsigned)gps.satellites.value());
  }

  // Check whether current scan has completed
  checkWiFiScan();

  // Start the next scan when ready
  if (!scanInProgress && now >= nextScanTime) {

    startWiFiScan();
  }
}
