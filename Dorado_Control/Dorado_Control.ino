#define lDir 0 //pinul 0 este pinul de direcţie pentru roata stângă
#define lSpeed 3 //pinul 3 este pinul cu care putem controla viteza motorului stâng
#define rSpeed 4 //pinul 4 este pinul cu care putem controla viteza motorului drept 
#define rDir 1 //pinul 1 este pinul de direcţie pentru roata dreaptă

#define maxPWM 255 //valoarea maximă a PWM-ului
#define minPWM 50 //valoarea mminimă a PWM-ului
#define motorForward LOW //valoarea logică a pinului de direcţie ca motorul să se rotească înainte
#define motorBackward HIGH //valoarea logică a pinului de direcţie ca motorul să se rotească înapoi

int incomingData = 0; //variabila de tip integer în care se memoriază caracterul recepţionat pe Bluetooth
int motorSpeed = maxPWM; //variabila de tip integer în care se memoriază viteza actuală a motoarelor, starea iniţială este 255.
int debug = 1; //variabila de tip integer în care activeayă funcţiile de debug
int turningSpeed= 200; //variabila de tip integer în care se memoriază viteya motoarelor pentru rotiriile spre dreapta si stânga
int serialEventDone = 0; //variabila de tip integer în care se memoriază dacă s-a recepţionat cevpe serial

void setup()
{
  pinMode(lSpeed, OUTPUT); //definirea pinului lSpeed ca iesire
  pinMode(rSpeed, OUTPUT); //definirea pinului rSpeed ca iesire
  pinMode(lDir, OUTPUT); //definirea pinului lDir ca iesire
  analogWrite(lSpeed, 255); //setarea PWM-ului la valoarea de 255 pentru lSpeed
  pinMode(rDir, OUTPUT); //definirea pinului rDir ca iesire
  analogWrite(rSpeed, 255); //setarea PWM-ului la valoarea de 255 pentru rSpeed
  Serial.begin(115200); //pornirea comunicării pe serial

  if(debug)
  {
    Serial.println("Serial Running.");
  };
}

void serialEvent()
{
  if(Serial.available()>0)
  {
    incomingData = Serial.read();	//citirea datelor recepţionate pe serial
  }
  serialEventDone = 1;
}

void loop()
{
  if(serialEventDone)
  {
    switch (incomingData)
    {
    case 'f':
      motorDrive(1);  //start moving forward
      if(debug)
      {
        Serial.println("received f");
      };
      break;

    case 'b':
      motorDrive(2);  //start moving backward
      if(debug)
      {
        Serial.println("received b") ;
      };
      break;

    case 'l':
      motorDrive(3);  //start turing left
      if(debug)
      {
        Serial.println("received l") ;
      };
      break;

    case 'r':
      motorDrive(4);  //start turning right
      if(debug)
      {
        Serial.println("received r") ;
      };
      break;

    case 's':
      motorDrive(0);  //stop the motors
      if(debug)
      {
        Serial.println("received s")  ;
      };
      break;
    default:
      break;
    }
    serialEventDone = 0;
  }

}
void motorDrive(int mode)
{
  switch (mode)
  {
  case 0:  //stop
    brake();

    if(debug)
    {
      Serial.println(motorSpeed)  ;
    };
    break;

  case 1:  //forward
    digitalWrite(lDir, motorForward);
    digitalWrite(rDir, motorForward);
    if(accelerate() == 1)
    {
      analogWrite(lSpeed, minPWM);
      analogWrite(rSpeed, minPWM);
    }
    if(debug)
    {
      Serial.println(motorSpeed)  ;
    };
    break;

  case 2:  //backward
    digitalWrite(lDir, motorBackward);
    digitalWrite(rDir, motorBackward);

    if(accelerate() == 1)
    {
      analogWrite(lSpeed, minPWM);
      analogWrite(rSpeed, minPWM);
    }
    if(debug)
    {
      Serial.println(motorSpeed)  ;
    };
    break;

  case 3:  //left
    digitalWrite(lDir, motorBackward);
    digitalWrite(rDir, motorForward);

    motorSpeed = turningSpeed;
    analogWrite(lSpeed, motorSpeed);
    analogWrite(rSpeed, motorSpeed);
    break;

  case 4:  //right
    digitalWrite(lDir, motorForward);
    digitalWrite(rDir, motorBackward);
    motorSpeed = turningSpeed;
    analogWrite(lSpeed, motorSpeed);
    analogWrite(rSpeed, motorSpeed);
    break;
  default:
    break;
  }
}
int brake()
{
  for(motorSpeed; motorSpeed<maxPWM; motorSpeed++)
  {
    analogWrite(lSpeed, motorSpeed);
    analogWrite(rSpeed, motorSpeed);
    delayMicroseconds(500);
  }
  digitalWrite(lSpeed, HIGH);
  digitalWrite(rSpeed, HIGH);

  if(debug)
  {
    Serial.println("brake 1")  ;
  };
  return 1;
}
int accelerate()
{
  for(motorSpeed = maxPWM; motorSpeed>minPWM; motorSpeed--)
  {
    analogWrite(lSpeed, motorSpeed);
    analogWrite(rSpeed, motorSpeed);
    delay(2);
  }

  if(debug)
  {
    Serial.println("accelerate 1")  ;
  };
  return 1;
}
