// ---------------------------------------------------------
// TESIS ANDRADE - FIRMWARE V13 (DIGITAL INTERRUPT + GSR)
// ---------------------------------------------------------

const int PIN_GSR = A2;
const int PIN_PPG_INT = 2; // Conexión obligatoria al Pin Digital 2 (INT0)
const int PIN_LED = 13;
const unsigned long SAMPLE_INTERVAL = 16; 

unsigned long lastSampleTime = 0;
bool isRunning = false;

// Variables volátiles para la memoria de la interrupción (ISR)
volatile unsigned long lastBeatTime = 0;
volatile unsigned long beatIntervalUs = 0;
volatile bool newBeatDetected = false;

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }
  
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_GSR, INPUT);
  pinMode(PIN_PPG_INT, INPUT); 
  
  // Secuencia de inicialización
  for(int i=0; i<3; i++) {
    digitalWrite(PIN_LED, HIGH); delay(100);
    digitalWrite(PIN_LED, LOW); delay(100);
  }
  
  // Vectorización de la interrupción en el flanco de subida
  attachInterrupt(digitalPinToInterrupt(PIN_PPG_INT), detectBeat, RISING);
  Serial.println("SYSTEM:READY_V3");
}

// ISR: Rutina de Servicio de Interrupción (Complejidad O(1))
void detectBeat() {
  unsigned long currentTime = micros();
  unsigned long interval = currentTime - lastBeatTime;
  
  // Filtro antirrebote hardware (Debounce biológico: máx 240 BPM / 250ms)
  if (interval > 250000UL) {
    beatIntervalUs = interval;
    lastBeatTime = currentTime;
    newBeatDetected = true;
  }
}

void loop() {
  // 1. Procesamiento de Comandos
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.replace("\r", "");
    
    if (cmd.length() > 0) {
      Serial.print("CMD_ECHO: ");
      Serial.println(cmd);
      
      if (cmd == "START" || cmd == "RESUME" || cmd == "SUBJECT_CHECK") {
        isRunning = true;
        digitalWrite(PIN_LED, HIGH);
        if (cmd == "SUBJECT_CHECK") Serial.println("STATUS:SUBJECT_CHECK_STARTED");
      }
      else if (cmd == "STOP" || cmd == "PAUSE") {
        isRunning = false;
        digitalWrite(PIN_LED, LOW);
      }
      else if (cmd == "STATUS") {
        Serial.println("SYSTEM:READY_V3");
      }
    }
  }

  // 2. Transmisión de Datos
  if (isRunning) {
    unsigned long currentMillis = millis();
    
    // Envío asíncrono del intervalo PPG (alta precisión)
    if (newBeatDetected) {
      noInterrupts(); // Bloqueo atómico de memoria
      unsigned long intervalToSend = beatIntervalUs;
      newBeatDetected = false;
      interrupts();
      
      Serial.print("PPG_INT:");
      Serial.println(intervalToSend);
    }
    
    // Envío sincrónico GSR (62.5 Hz)
    if (currentMillis - lastSampleTime >= SAMPLE_INTERVAL) {
      lastSampleTime = currentMillis;
      Serial.print("GSR:");
      Serial.println(analogRead(PIN_GSR));
    }
  }
}