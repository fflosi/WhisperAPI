using System;
using System.IO;
using System.Text;
using Whisper;

class WhisperPipe
{
    static Model model;
    static Context context;
    static iMediaFoundation mf;

    static void Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("Usage: WhisperPipe <model_path>");
            Environment.Exit(1);
        }
        string modelPath = args[0];
        try
        {
            model = Library.loadModel(modelPath);
            context = model.createContext();
            mf = Library.initMediaFoundation();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Failed to load model: {ex.Message}");
            Environment.Exit(2);
        }

        var stdin = Console.OpenStandardInput();
        var stdout = Console.OpenStandardOutput();
        var lenBuf = new byte[4];
        while (true)
        {
            // Read length
            int read = stdin.Read(lenBuf, 0, 4);
            if (read == 0) break; // EOF
            if (read < 4) throw new EndOfStreamException();
            int length = BitConverter.ToInt32(lenBuf, 0);
            if (length <= 0) break;
            byte[] buffer = new byte[length];
            int offset = 0;
            while (offset < length)
            {
                int r = stdin.Read(buffer, offset, length - offset);
                if (r <= 0) throw new EndOfStreamException();
                offset += r;
            }
            // Decode WAV
            using var audioReader = mf.loadAudioFileData(buffer, buffer.Length, false);
            context.runFull(audioReader, null, null);
            var result = context.results();
            string text = string.Join("\n", result.segments.Select(seg => seg.text));
            byte[] textBytes = Encoding.UTF8.GetBytes(text);
            byte[] outLen = BitConverter.GetBytes(textBytes.Length);
            stdout.Write(outLen, 0, 4);
            stdout.Write(textBytes, 0, textBytes.Length);
            stdout.Flush();
        }
    }
}
