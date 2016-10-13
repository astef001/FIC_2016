using System;
using System.Collections.Generic;
using System.IO.Ports;
using System.Linq;
using System.Text;
using System.Threading;

public class SerialTest
{
    public static void Main()
    {

        try
        {
            String[] arguments = Environment.GetCommandLineArgs();
            String comName = arguments[1];
            Console.WriteLine("Opening port " + comName);

            SerialPort serialPort = new SerialPort();
            serialPort.BaudRate = 115200;
            //serialPort.PortName = "COM4"; // Set in Windows
            serialPort.PortName = comName;
            serialPort.Open();



            while (serialPort.IsOpen)
            {

                /*
                char c = Console.ReadKey().KeyChar;
                Console.Write(c);
                serialPort.Write(c.ToString());
                 */

                serialPort.Write("l");
                Thread.Sleep(1000);
                serialPort.Write("s");


            }
        }
        catch (Exception e)
        {
            
        }
    }
}