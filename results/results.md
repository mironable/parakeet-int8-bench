# Results

Machine: Apple M1 Pro, 16.0 GB, macOS-27.0.1-arm64-arm-64bit. Batch size 1, 4 CPU threads, onnxruntime 1.31.0.

## Accuracy and latency

| Dataset | Clips | Variant | WER % | p50 ms | p95 ms | RTF | Identical tokens | WER vs reference % |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| edacc | 300 | PyTorch fp32 | 14.82 | 765 | 1197 | 0.1218 | - | - |
| edacc | 300 | ONNX fp32 | 14.82 | 269 | 1111 | 0.0588 | 100.0% vs PyTorch fp32 | 0.00 |
| edacc | 300 | ONNX int8 | 14.73 | 133 | 542 | 0.0286 | 74.7% vs ONNX fp32 | 1.64 |
| fleurs_de | 300 | PyTorch fp32 | 6.31 | 1251 | 1525 | 0.0976 | - | - |
| fleurs_de | 300 | ONNX fp32 | 6.31 | 649 | 1276 | 0.0553 | 100.0% vs PyTorch fp32 | 0.00 |
| fleurs_de | 300 | ONNX int8 | 6.42 | 312 | 614 | 0.0267 | 81.7% vs ONNX fp32 | 0.94 |
| librispeech | 300 | PyTorch fp32 | 1.93 | 788 | 1070 | 0.1119 | - | - |
| librispeech | 300 | ONNX fp32 | 1.93 | 345 | 988 | 0.0586 | 100.0% vs PyTorch fp32 | 0.00 |
| librispeech | 300 | ONNX int8 | 1.93 | 163 | 469 | 0.0279 | 88.0% vs ONNX fp32 | 0.27 |

## Size and memory

| Variant | Weights on disk MB | Peak RSS MB | Load s |
|---|---:|---:|---:|
| PyTorch fp32 | 2431.6 | 5413 | 19.9 |
| ONNX fp32 | 2431.6 | 3070 | 5.1 |
| ONNX int8 | 891.9 | 1886 | 1.1 |

## EdAcc by first language (groups with at least 20 clips)

| L1 | Clips | WER fp32 % | WER int8 % | Change |
|---|---:|---:|---:|---:|
| Catalan | 85 | 19.70 | 19.88 | +0.18 |
| Scottish English | 70 | 15.77 | 15.51 | -0.26 |
| Spanish | 38 | 15.97 | 15.56 | -0.42 |
| Lithuanian | 28 | 9.05 | 8.62 | -0.43 |
| Sinhalese | 30 | 13.29 | 12.82 | -0.47 |

## Where int8 changed the transcript most: edacc

- edacc_00655
  - ref:  is okay
  - fp32: con reunion gratuita terminaba en diez minutos ok
  - int8: con reunion gratuita terminaba en diez minutos ok ah
- edacc_00868
  - ref:  0 i saw i saw a flight change as well okay but no no that is personal we cannot say
  - fp32: 0 i saw i saw flight change as well okay but no no that is personal we cannot say it in that
  - int8: 0 it is so it saw flight change as well okay but no no that is personal we cannot say it in that
- edacc_00747
  - ref:  it was my naughty pleasure as well
  - fp32: it was my naughty
  - int8: it was my nutty
- edacc_00147
  - ref:  yeah i do not think lamb is a
  - fp32: yeah i do not think line incidental yeah
  - int8: yeah i do not think we are incidental yeah
- edacc_00270
  - ref:  why what are the most special things about kandy
  - fp32: why what are the most special things about candy
  - int8: by what are the most special things about candy

## Where int8 changed the transcript most: fleurs_de

- fleurs_de_12203116308364043833
  - ref:  eine internetsuche nach kurs über feindselige umgebungen wird vermutlich die adresse eine örtlichen firma anzeigen
  - fp32: eine internetsuche nach kurs über feindselige umgebungen wird vermutlich die adresse einer örtlichen firma anzeigen
  - int8: eine internetsuche nach kursüber feindselige umgebungen wird vermutlich die adresse einer örtlichen firma anzeigen
- fleurs_de_5170221231862717273
  - ref:  der kongress begann im august 2005 mit der finanzierung der anti obszönitätsinitiative und legte fest dass das fbi 10 agenten für die erwachsenenpornographie einsetzen muss
  - fp32: der kongress begann im august 2005 mit der finanzierung der anti obszunitätsinitiative und legte fest dass das fbi zehn agenten für die erwachsenenpornografie einsetzen muss
  - int8: der kongress begann im august zwei tausend fünf mit der finanzierung der anti obszunitätsinitiative und legte fest dass das fbi zehn agenten für die erwachsenenpornografie einsetzen muss
- fleurs_de_16580977694186453435
  - ref:  antike kulturen und volksstämme begannen sie als nutztiere zu halten um leicht an deren milch fell fleisch und haut zu kommen
  - fp32: antike kulturen und volksstämme begannen sie als nutztiere zu halten um leicht an deren milch fell fleisch und haut zu kommen
  - int8: antikulturen und volksstämme begannen sie als nutztiere zu halten um leicht an deren milch fell fleisch und haut zu kommen
- fleurs_de_10844590730633377222
  - ref:  ihr thermisches verhalten ist nicht so gleichmäßig wie das großer höhlen auf der erde die oft eine ziemlich konstante temperatur haben aber es stimmt damit überein dass es sich um tiefe löcher im boden handelt sagte glen cushing vom astrogeologie team des united states geological survey und der northern arizona university in flagstaff arizona
  - fp32: ihr thermisches verhalten ist nicht so gleichmäßig wie das großer höhlen auf der erde die oft eine ziemlich konstante temperatur haben aber es stimmt damit überein dass es sich um tiefe löcher im boden handelt sagt glen cushing vom astrogeologie team des united states geological survey usgs und der northern arizona university in flagstaff arizona
  - int8: ihr thermisches verhalten ist nicht so gleichmäßig wie das großer höhlen auf der erde die oft eine ziemlich konstante temperatur haben aber es stimmt damit überein dass es sich um tiefe löcher am boden handelt sagt glenn cushing vom astro geologie team des united states geological survey usgs und der northern arizona university in flagstaff arizona
- fleurs_de_5085509947417478932
  - ref:  lakkha singh präsentierte den chhappan bhog bhajan ebenfalls der sänger raju khandelwal begleitete ihn
  - fp32: lakas singh präsentierte den chappan bok bajan ebenfalls der sänger rajut kandal begleitete ihn
  - int8: lager sh präsentierte den chapan bok bhagan ebenfalls der sänger rajut kandal begleitete ihn

## Where int8 changed the transcript most: librispeech

- 1580-141083-0049
  - ref:  anyone in the room could get out yes sir
  - fp32: anyone in the room could get out yes sir
  - int8: any one in the room could get out yes sir
- 4446-2273-0016
  - ref:  her hair is still like flax and her blue eyes are just like a baby is and she has the same 3 freckles on her little nose and talks about going back to her bains de mer
  - fp32: her hair is still like flax and her blue eyes are just like a baby is and she has the same 3 freckles on her little nose and talks about going back to her band de mer
  - int8: her hair is still like flax and her blue eyes are just like a baby is and she has the same 3 freckles on her little nose and talks about going back to her bandamer
- 7127-75947-0027
  - ref:  to say nothing said montalais so that when mademoiselle de tonnay charente thinks athenais is the only one who knows it
  - fp32: to say nothing said montalay so that when mademoiselle de tonetchirant thinks athenay is the only one who knows it
  - int8: to say nothing said montalay so that when mademoiselle d atonetchirant thinks athenay is the only one who knows it
