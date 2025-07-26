using System;
using System.Runtime.InteropServices;
using Whisper;

public class PCMBuffer : iAudioBuffer
{
    private GCHandle handle;
    private IntPtr pcmPtr;
    private int sampleCount;
    private TimeSpan startTime;

    public PCMBuffer(float[] pcm, TimeSpan? start = null)
    {
        sampleCount = pcm.Length;
        handle = GCHandle.Alloc(pcm, GCHandleType.Pinned);
        pcmPtr = handle.AddrOfPinnedObject();
        startTime = start ?? TimeSpan.Zero;
    }

    public int countSamples() => sampleCount;
    public IntPtr getPcmMono() => pcmPtr;
    public IntPtr getPcmStereo() => IntPtr.Zero;
    public void getTime(out TimeSpan time) => time = startTime;

    public void Dispose()
    {
        if (handle.IsAllocated)
            handle.Free();
    }
}
