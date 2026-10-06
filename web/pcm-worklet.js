// 采集端 AudioWorklet：麦克风 float32（context 采样率）→ 16kHz Int16，
// 攒够 512 样本（32ms）发一帧给主线程。抽樣用线性抽取，语音场景足够。
class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const inRate = (options.processorOptions && options.processorOptions.sampleRate) || sampleRate;
    this.ratio = inRate / 16000;
    this.acc = 0;
    this.buf = [];
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        this.buf.push(ch[i]);
      }
    }
    while (this.buf.length >= 512) {
      const frame = this.buf.splice(0, 512);
      const i16 = new Int16Array(512);
      for (let j = 0; j < 512; j++) {
        const v = Math.max(-1, Math.min(1, frame[j]));
        i16[j] = v < 0 ? v * 32768 : v * 32767;
      }
      this.port.postMessage(i16.buffer, [i16.buffer]);
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCapture);
