using System;
using System.IO;
using System.Text;
using System.Linq;
using Whisper;

class PipeParams
{
    public string Language = "pt";
    public bool Translate = false;
    public int CpuThreads = 4;
    public int MaxTextCtx = 0;
    public int OffsetMs = 0;
    public int DurationMs = 0;
    public bool PrintTimestamps = true;
    public bool PrintSpecial = false;
    public bool SpeedUpAudio = false;
    public int MaxLen = 0;
    public float WordThold = 0.01f;
    // Add more as needed

    public static PipeParams Parse(string[] args)
    {
        var p = new PipeParams();
        for (int i = 1; i < args.Length; ++i)
        {
            switch (args[i])
            {
                case "--language":
                    p.Language = args[++i];
                    break;
                case "--translate":
                    p.Translate = true;
                    break;
                case "--threads":
                    p.CpuThreads = int.Parse(args[++i]);
                    break;
                case "--max-context":
                    p.MaxTextCtx = int.Parse(args[++i]);
                    break;
                case "--offset-ms":
                    p.OffsetMs = int.Parse(args[++i]);
                    break;
                case "--duration-ms":
                    p.DurationMs = int.Parse(args[++i]);
                    break;
                case "--print-timestamps":
                    p.PrintTimestamps = true;
                    break;
                case "--no-timestamps":
                    p.PrintTimestamps = false;
                    break;
                case "--print-special":
                    p.PrintSpecial = true;
                    break;
                case "--speed-up":
                    p.SpeedUpAudio = true;
                    break;
                case "--max-len":
                    p.MaxLen = int.Parse(args[++i]);
                    break;
                case "--word-thold":
                    p.WordThold = float.Parse(args[++i]);
                    break;
                // Add more options as needed
            }
        }
        return p;
    }

    public void Apply(ref Whisper.Parameters parameters)
    {
        var langKey = Whisper.Library.languageFromCode(Language);
        if (langKey.HasValue)
        {
            parameters.language = langKey.Value;
            Console.Error.WriteLine($"Set language to: {Language}");
        }
        else
        {
            Console.Error.WriteLine($"Warning: Language code '{Language}' not recognized. Using model default.");
        }
        parameters.cpuThreads = CpuThreads;
        parameters.n_max_text_ctx = MaxTextCtx;
        parameters.offset_ms = OffsetMs;
        parameters.duration_ms = DurationMs;
        parameters.setFlag(Whisper.eFullParamsFlags.PrintTimestamps, PrintTimestamps);
        parameters.setFlag(Whisper.eFullParamsFlags.PrintSpecial, PrintSpecial);
        parameters.setFlag(Whisper.eFullParamsFlags.Translate, Translate);
        parameters.setFlag(Whisper.eFullParamsFlags.SpeedupAudio, SpeedUpAudio);
        parameters.max_len = MaxLen;
        parameters.thold_pt = WordThold;
        // Add more parameter mappings as needed
    }
}

class WhisperPipe
{
    static iModel model;
    static Context context;

    static void Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("Usage: WhisperPipe <model_path> [options]");
            Environment.Exit(1);
        }
        string modelPath = args[0];
        var pipeParams = PipeParams.Parse(args);
        try
        {
            // Use GPU by default, with recommended flags and adapter
            model = Library.loadModel(modelPath, Whisper.eGpuModelFlags.None, null, Whisper.eModelImplementation.GPU);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Failed to load model: {ex.Message}");
            Environment.Exit(2);
        }

        var stdin = Console.OpenStandardInput();
        var stdout = Console.OpenStandardOutput();
        var lenBuf = new byte[4];
        //Console.Error.WriteLine($"****Starting:*****");
        //Console.Error.Flush();
        while (true)
        {
            int read = stdin.Read(lenBuf, 0, 4);
            if (read == 0) break;
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
            // Convert byte[] to float[] (little-endian float32 PCM)
            int floatCount = buffer.Length / 4;
            float[] pcm = new float[floatCount];
            for (int i = 0; i < floatCount; ++i)
                pcm[i] = BitConverter.ToSingle(buffer, i * 4);

            // Re-create context for each buffer and dispose automatically
            using var context = model.createContext();
            // Set parameters before inference
            if (!model.isMultilingual())
            {
                Console.Error.WriteLine($"Warning: Model is not multilingual. Language parameter will be ignored.");
            }
            pipeParams.Apply(ref context.parameters);
            // Debug: print all parameter values before inference
            var parameters = context.parameters;
            Console.Error.WriteLine($"Parameters: Language={parameters.language}, CpuThreads={parameters.cpuThreads}, MaxTextCtx={parameters.n_max_text_ctx}, OffsetMs={parameters.offset_ms}, DurationMs={parameters.duration_ms}, PrintTimestamps={pipeParams.PrintTimestamps}, PrintSpecial={pipeParams.PrintSpecial}, Translate={pipeParams.Translate}, SpeedUpAudio={pipeParams.SpeedUpAudio}, MaxLen={pipeParams.MaxLen}, WordThold={pipeParams.WordThold}");
            using var audioBuffer = new PCMBuffer(pcm);
            context.runFull(audioBuffer, null);
            var result = context.results();
            var segments = result.segments.ToArray();
            string text = string.Join("\n", segments.Select(seg => seg.text));
            byte[] textBytes = Encoding.UTF8.GetBytes(text);
            byte[] outLen = BitConverter.GetBytes(textBytes.Length);
            stdout.Write(outLen, 0, 4);
            stdout.Write(textBytes, 0, textBytes.Length);
            stdout.Flush();
        }
    }
}
