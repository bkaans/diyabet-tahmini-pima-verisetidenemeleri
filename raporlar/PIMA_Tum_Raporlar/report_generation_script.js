const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, AlignmentType, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, LevelFormat, PageBreak
} = require('docx');
const fs = require('fs');
const path = require('path');

const REPORT_DIR = __dirname;
const CHART_DIR = path.join(REPORT_DIR, 'charts');
const chartPath = (fileName) => path.join(CHART_DIR, fileName);
const reportPath = (fileName) => path.join(REPORT_DIR, fileName);

// ─── Renk paleti ─────────────────────────────────────────────────────────
const DARK="1B4332",MID="2D6A4F",LIGHT="74C69D",PALE="D8F3DC",WHITE="FFFFFF";
const LGRAY="F4F6F6",MGRAY="BDC3C7",TEXT="1A1A2E",RED="C0392B",AMBER="D35400",GOLD="F39C12";

// ─── Yardımcılar ─────────────────────────────────────────────────────────
const noBorder={style:BorderStyle.NIL,size:0,color:"FFFFFF"};
const noAll={top:noBorder,bottom:noBorder,left:noBorder,right:noBorder};
function bdr(c="CCCCCC",s=4){return{style:BorderStyle.SINGLE,size:s,color:c};}
function allB(c="CCCCCC"){return{top:bdr(c),bottom:bdr(c),left:bdr(c),right:bdr(c)};}

function cell(txt,{w=2340,fill=WHITE,bold=false,color=TEXT,align=AlignmentType.LEFT,
  vAlign=VerticalAlign.CENTER,borders=allB(),size=19,italic=false}={}){
  return new TableCell({
    width:{size:w,type:WidthType.DXA},
    shading:{fill,type:ShadingType.CLEAR},
    borders,verticalAlign:vAlign,
    margins:{top:90,bottom:90,left:130,right:130},
    children:Array.isArray(txt)?txt:[new Paragraph({alignment:align,
      children:[new TextRun({text:String(txt),bold,color,size,font:"Arial",italics:italic})]})]
  });
}
function hCell(txt,w=2340,size=18){
  return cell(txt,{w,fill:DARK,bold:true,color:WHITE,borders:allB(DARK),size});
}
function sp(n=140){return new Paragraph({spacing:{after:n},children:[]});}

function heading(txt,level=1){
  const sizes={1:32,2:26,3:22};
  const colors={1:DARK,2:MID,3:MID};
  return new Paragraph({
    spacing:{before:level===1?300:220,after:130},
    border:level===1?{bottom:{style:BorderStyle.SINGLE,size:6,color:MID,space:4}}:undefined,
    children:[new TextRun({text:txt,bold:true,size:sizes[level],color:colors[level],font:"Arial"})]
  });
}
function para(txt,{bold=false,color=TEXT,size=19,italic=false,spacing={after:110}}={}){
  return new Paragraph({spacing,children:[new TextRun({text:txt,bold,color,size,font:"Arial",italics:italic})]});
}
function noteBox(txt){
  return new Table({
    width:{size:9840,type:WidthType.DXA},columnWidths:[9840],
    rows:[new TableRow({children:[new TableCell({
      width:{size:9840,type:WidthType.DXA},
      shading:{fill:PALE,type:ShadingType.CLEAR},
      borders:{left:{style:BorderStyle.SINGLE,size:18,color:MID},top:noBorder,bottom:noBorder,right:noBorder},
      margins:{top:100,bottom:100,left:180,right:140},
      children:[new Paragraph({children:[new TextRun({text:txt,italics:true,size:18,color:MID,font:"Arial"})]})]
    })]})],
  });
}
function img(path,wPx=656,hPx=292){
  const buf=fs.readFileSync(path);
  return new Paragraph({
    spacing:{before:160,after:200},alignment:AlignmentType.CENTER,
    children:[new ImageRun({data:buf,transformation:{width:wPx,height:hPx},type:"png"})]
  });
}
function summaryTable(rows,col1=3600,col2=6240){
  return new Table({
    width:{size:9840,type:WidthType.DXA},columnWidths:[col1,col2],
    rows:rows.map(([k,v],i)=>new TableRow({children:[
      cell(k,{w:col1,fill:i%2===0?LGRAY:WHITE,bold:true,color:DARK,size:19}),
      cell(v,{w:col2,fill:i%2===0?LGRAY:WHITE,color:TEXT,size:19}),
    ]}))
  });
}
function coverBlock(title,subtitle,tag){
  return new Table({
    width:{size:9840,type:WidthType.DXA},columnWidths:[9840],
    rows:[new TableRow({children:[new TableCell({
      width:{size:9840,type:WidthType.DXA},
      shading:{fill:DARK,type:ShadingType.CLEAR},borders:noAll,
      margins:{top:400,bottom:400,left:400,right:400},
      children:[
        new Paragraph({alignment:AlignmentType.LEFT,spacing:{after:80},
          children:[new TextRun({text:tag,size:21,color:LIGHT,font:"Arial"})]}),
        new Paragraph({alignment:AlignmentType.LEFT,spacing:{after:100},
          children:[new TextRun({text:title,bold:true,size:36,color:WHITE,font:"Arial"})]}),
        new Paragraph({alignment:AlignmentType.LEFT,spacing:{after:0},
          children:[new TextRun({text:subtitle,size:18,color:LIGHT,font:"Arial",italics:true})]}),
      ]
    })]})],
  });
}
function makeHeader(left,right){
  return new Header({children:[new Table({
    width:{size:9840,type:WidthType.DXA},columnWidths:[6840,3000],
    rows:[new TableRow({children:[
      new TableCell({width:{size:6840,type:WidthType.DXA},
        borders:{...noAll,bottom:{style:BorderStyle.SINGLE,size:4,color:MID}},
        margins:{bottom:80},children:[new Paragraph({children:[
          new TextRun({text:left,bold:true,size:17,color:DARK,font:"Arial"})]})]
      }),
      new TableCell({width:{size:3000,type:WidthType.DXA},
        borders:{...noAll,bottom:{style:BorderStyle.SINGLE,size:4,color:MID}},
        margins:{bottom:80},children:[new Paragraph({alignment:AlignmentType.RIGHT,
          children:[new TextRun({text:right,size:17,color:MGRAY,font:"Arial"})]})
        ]
      }),
    ]})]
  })]});
}
function leakageTable(rows){
  return new Table({
    width:{size:9840,type:WidthType.DXA},columnWidths:[3600,2520,1560,2160],
    rows:[
      new TableRow({tableHeader:true,children:[
        hCell("Kontrol Adımı",3600),hCell("Beklenen",2520),hCell("Sonuç",1560),hCell("Açıklama",2160),
      ]}),
      ...rows.map(([k,b,s,a],i)=>new TableRow({children:[
        cell(k,{w:3600,fill:i%2?WHITE:LGRAY,bold:true,color:DARK}),
        cell(b,{w:2520,fill:i%2?WHITE:LGRAY,color:TEXT}),
        cell(s,{w:1560,fill:s==="✓"||s==="0"||s==="Evet"||s==="Hayır"?PALE:LGRAY,
                bold:true,color:s==="✓"||s==="0"||s==="Evet"||s==="Hayır"?MID:TEXT,
                align:AlignmentType.CENTER}),
        cell(a,{w:2160,fill:i%2?WHITE:LGRAY,color:TEXT,size:18}),
      ]}))
    ]
  });
}
function pageProps(){
  return {page:{size:{width:12240,height:15840},margin:{top:1080,right:1200,bottom:1080,left:1200}}};
}

// ═══════════════════════════════════════════════════════════════════════════
//  RAPOR 1
// ═══════════════════════════════════════════════════════════════════════════
async function buildR1(){
const doc=new Document({
  styles:{default:{document:{run:{font:"Arial",size:19,color:TEXT}}}},
  sections:[{
    properties:pageProps(),
    headers:{default:makeHeader("PIMA Veriseti · Veri Arttırmadan Deneme Benchmarkları","Benchmark Raporu · 2025")},
    children:[
      coverBlock(
        "PIMA Veriseti Veri Arttırmadan Deneme Benchmarkları",
        "StratifiedKFold · Threshold Tuning · Ensemble · Literatür Profilleri",
        "PIMA Indians Diabetes"
      ),
      sp(200),

      heading("Veri Seti ve Çalışma Özeti"),
      summaryTable([
        ["Veri Seti","PIMA Indians Diabetes"],
        ["Toplam Gözlem","768 satır"],
        ["Sınıf Dağılımı","500 negatif, 268 pozitif"],
        ["Veri Bütünlüğü","Ham CSV dosyasına satır ekleme, silme veya sentetik veri uygulanmadı"],
        ["Önceki Deploy Modeli","XGBoost — Accuracy %77.3, ROC-AUC %82.6, F1 %72.4"],
        ["CV Stratejisi","StratifiedKFold, 10-fold; Repeated CV; Threshold Tuning"],
        ["En İyi Tek Split","SVM RBF (Seed 46) — Accuracy %81.8, ROC-AUC %85.4"],
        ["En İyi Final Artifact","SVM RBF — Accuracy %79.2, ROC-AUC %84.5"],
      ]),

      sp(240),
      heading("1. Keşifsel Veri Analizi (EDA)"),
      para("Sınıf dağılımının dengesiz olduğu ve bazı klinik değişkenlerde 0 değerlerinin yoğunlaştığı saptandı. "+
           "Özellikle Insulin (374 adet) ve SkinThickness değişkenlerindeki 0 değerleri ham veriyle çalışırken "+
           "modelin öğrenmesini zorlaştıran önemli veri kalitesi unsurlarıdır."),

      new Table({
        width:{size:9840,type:WidthType.DXA},columnWidths:[2800,2000,2340,2700],
        rows:[
          new TableRow({tableHeader:true,children:[
            hCell("Değişken",2800),hCell("Outcome Korelasyonu",2000),
            hCell("Sınıf Ort. Farkı",2340),hCell("0 Değeri Sayısı",2700)]}),
          ...([
            ["Glucose","0.467","31.28","5",false],
            ["BMI","0.293","4.84","11",false],
            ["Age","0.238","5.88","0",false],
            ["Pregnancies","0.222","1.57","111",true],
            ["DiabetesPedigreeFunction","0.174","0.12","0",false],
            ["Insulin","0.131","31.54","374",true],
          ]).map(([v,k,o,s,warn],i)=>new TableRow({children:[
            cell(v,{w:2800,fill:i%2?WHITE:LGRAY,bold:true,color:DARK}),
            cell(k,{w:2000,fill:i%2?WHITE:LGRAY,align:AlignmentType.CENTER}),
            cell(o,{w:2340,fill:i%2?WHITE:LGRAY,align:AlignmentType.CENTER}),
            cell(s,{w:2700,fill:warn?"#FEF9E7":i%2?WHITE:LGRAY,bold:warn,
                    color:warn?AMBER:TEXT,align:AlignmentType.CENTER}),
          ]}))
        ]
      }),
      sp(160),
      img(chartPath("r1_sifir_degeri.png"),656,268),

      sp(260),
      heading("2. Veri Arttırmadan Model Karşılaştırması"),
      para("Önceki projedeki uygulama modeli XGBoost tabanlıydı. Aynı veri sınırı korunarak "+
           "StratifiedKFold, threshold tuning, class_weight, ensemble ve farklı model aileleri denendi. "+
           "En yüksek görünen Accuracy değeri agresif aramadaki tek split sonucunda %81.8 oldu; "+
           "ancak final artifact olarak tekrar eğitildiğinde %79.2'ye geriledi."),

      new Table({
        width:{size:9840,type:WidthType.DXA},columnWidths:[2800,2200,1200,1200,1200,1240],
        rows:[
          new TableRow({tableHeader:true,children:[
            hCell("Deney Grubu",2800),hCell("Model",2200),
            hCell("Accuracy",1200),hCell("ROC-AUC",1200),hCell("F1",1200),hCell("Not",1240)]}),
          ...([
            ["Önceki uygulama modeli","XGBoost","%77.3","%82.6","%72.4","Eski deploy",false],
            ["Önceki veri arttırmadan opt.","XGBoost (no SMOTE)","%77.3","%82.6","%72.4","SMOTE yok",false],
            ["Yeni hızlı arama finali","stacking_grid_refined","%74.0","%82.0","%55.6","Stacking",false],
            ["Agresif arama — en iyi split","SVM RBF","%81.8","%85.4","%71.4","Seed 46",true],
            ["Agresif arama — final artifact","SVM RBF","%79.2","%84.5","%66.7","Tekrar eğitim",false],
          ]).map(([d,m,a,r,f,n,sel],i)=>new TableRow({children:[
            cell(d,{w:2800,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT}),
            cell(m,{w:2200,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT}),
            cell(a,{w:1200,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER}),
            cell(r,{w:1200,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER}),
            cell(f,{w:1200,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER}),
            cell(n,{w:1240,fill:sel?PALE:i%2?WHITE:LGRAY,color:TEXT,size:18}),
          ]}))
        ]
      }),
      sp(160),
      img(chartPath("r1_model_comparison.png"),656,300),

      sp(260),
      heading("3. Literatür Profillerinden Üretilen Deneyler"),
      para("Hossain çizgisinde KNN ve LightGBM, Amma çizgisinde RF + RBF-SVM + KNN voting, "+
           "Altamimi çizgisinde KNNImputer + XGB + RF + ExtraTrees soft voting, "+
           "Ansari çizgisinde ise geniş C/gamma aramalı RBF-SVM denendi."),

      new Table({
        width:{size:9840,type:WidthType.DXA},columnWidths:[2800,2040,960,1200,1320,1520],
        rows:[
          new TableRow({tableHeader:true,children:[
            hCell("Deney",2800),hCell("Literatür Referansı",2040),hCell("CV",960),
            hCell("CV Accuracy",1200),hCell("Holdout Acc.",1320),hCell("Not",1520)]}),
          ...([
            ["Ansari SVM RBF Geniş Tuning","Ansari et al. 2025","10-fold","%79.0","%70.1","En iyi CV"],
            ["Ansari SVM RBF Feature Sel.","Ansari et al. 2025","10-fold","%78.7","%72.7",""],
            ["Amma RF+RBF-SVM+KNN Hard","Amma N.G. 2024","10-fold","%77.0","%74.7","En iyi holdout (2.)"],
            ["Amma RF+RBF-SVM+KNN Soft","Amma N.G. 2024","5-fold","%76.7","%73.4",""],
            ["Altamimi KNNImputer+XGB+RF+ET Soft","Altamimi et al. 2024","5-fold","%76.4","%72.7",""],
            ["Hossain LightGBM+KNN+AdaBoost Soft","Hossain et al. 2022","5-fold","%75.6","%76.6","En iyi holdout"],
          ]).map(([d,r,c,ca,ha,n],i)=>new TableRow({children:[
            cell(d,{w:2800,fill:i%2?WHITE:LGRAY,bold:true,color:DARK,size:18}),
            cell(r,{w:2040,fill:i%2?WHITE:LGRAY,color:TEXT,size:18}),
            cell(c,{w:960,fill:i%2?WHITE:LGRAY,color:TEXT,align:AlignmentType.CENTER}),
            cell(ca,{w:1200,fill:i%2?WHITE:LGRAY,color:TEXT,align:AlignmentType.CENTER}),
            cell(ha,{w:1320,fill:i%2?WHITE:LGRAY,bold:true,color:MID,align:AlignmentType.CENTER}),
            cell(n,{w:1520,fill:i%2?WHITE:LGRAY,color:TEXT,size:17,italic:true}),
          ]}))
        ]
      }),
      sp(160),
      img(chartPath("r1_literatur.png"),656,280),

      sp(260),
      heading("4. Genel Sonuç"),
      noteBox(
        "Ham veri dosyası korunarak yapılan denemelerde %90 ve üzeri Accuracy hedefine ulaşılamadı. "+
        "En iyi literatür tabanlı CV sonucu Ansari benzeri SVM profilinde %79.0 Accuracy verdi; "+
        "en iyi holdout sonucu ise Hossain benzeri LightGBM + KNN + AdaBoost voting profilinde %76.6 oldu. "+
        "Çok yüksek literatür skorlarının büyük çoğunluğu ağır preprocessing, resampling, "+
        "feature selection veya sentetik veri üretimi ile ilişkili görünmektedir."
      ),
      sp(80),
    ]
  }]
});
const buf=await Packer.toBuffer(doc);
fs.writeFileSync(reportPath("01_raw_pima_benchmark_report.docx"),buf);
console.log("Rapor 1 tamamlandı.");
}

// ═══════════════════════════════════════════════════════════════════════════
//  RAPOR 2
// ═══════════════════════════════════════════════════════════════════════════
async function buildR2(){
const doc=new Document({
  styles:{default:{document:{run:{font:"Arial",size:19,color:TEXT}}}},
  sections:[{
    properties:pageProps(),
    headers:{default:makeHeader("Source ID Kontrollü PIMA + Sentetik Benchmark Raporu","Leakage Kontrolü · 2025")},
    children:[
      coverBlock(
        "Source ID Kontrollü ve PIMA + Sentetik Veri İçeren Benchmark Raporu",
        "Leakage Kontrolü · Source ID Aile Yapısı · Cross-Validation",
        "PIMA Indians Diabetes"
      ),
      sp(200),

      heading("Çalışma Özeti"),
      summaryTable([
        ["Orijinal Veri","768 satır"],
        ["Orijinal Dağılım","500 negatif, 268 pozitif"],
        ["Geliştirme Verisi (Dev)","614 satır"],
        ["External Holdout","154 satır"],
        ["Final Benchmark","PIMA + Sentetik 2700/sınıf"],
        ["Final Model","XGBoost — SkinThickness çıkarıldı"],
        ["Holdout Doğruluk","%81.25"],
        ["Holdout ROC-AUC","%89.14"],
        ["Sızıntı Durumu","Temiz — tüm source_id kontrolleri geçildi"],
      ]),

      sp(240),
      heading("1. Leakage Kontrolü ve Source ID Aile Yapısı"),
      para("Sentetik veri üretiminde her orijinal PIMA satırı bir kaynak aile olarak kabul edilmiştir. "+
           "Her orijinal satıra original_{index} formatında bir source_id atanmış; bu satırdan üretilen "+
           "tüm sentetik örnekler aynı source_id değerini taşımıştır. Böylece aynı kaynak aileden gelen "+
           "örneklerin hem eğitim hem de test tarafına düşmesi engellenmiştir."),

      leakageTable([
        ["Train/Test source_id kesişimi","0","0","Eğitim-test sızıntısı yok"],
        ["CV fold source_id kesişimi","0","0","Çapraz doğrulama temiz"],
        ["Exact duplicate","0","0","Birebir kopya yok"],
        ["Near duplicate oranı","0'a yakın","0.0000","Aşırı benzerlik yok"],
        ["Minimum mesafe","Raporlanır","0.0968","Örnekler yeterince ayrışık"],
        ["External holdout izolasyonu","Evet","✓","Üretimden önce ayrıldı"],
        ["Bağımsız sentetik source_id","Hayır","✓","Tüm örnekler dev ailesine bağlı"],
      ]),

      sp(240),
      heading("2. Sentetik Veri Adayı Boyutları"),

      new Table({
        width:{size:9840,type:WidthType.DXA},columnWidths:[2400,1680,1680,1680,2400],
        rows:[
          new TableRow({tableHeader:true,children:[
            hCell("Veri Adayı",2400),hCell("Orijinal",1680),hCell("Sentetik",1680),
            hCell("Toplam",1680),hCell("Dağılım",2400)]}),
          ...([
            ["PIMA Orijinal","614","0","614","400 negatif, 214 pozitif",false],
            ["PIMA + Sentetik 2500","614","4.386","5.000","2.500 / 2.500",false],
            ["PIMA + Sentetik 2700","614","4.786","5.400","2.700 / 2.700",true],
            ["PIMA + Sentetik 5000","614","9.386","10.000","5.000 / 5.000",false],
          ]).map(([a,o,s,t,d,sel],i)=>new TableRow({children:[
            cell(a,{w:2400,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT}),
            cell(o,{w:1680,fill:sel?PALE:i%2?WHITE:LGRAY,align:AlignmentType.CENTER}),
            cell(s,{w:1680,fill:sel?PALE:i%2?WHITE:LGRAY,align:AlignmentType.CENTER}),
            cell(t,{w:1680,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER}),
            cell(d,{w:2400,fill:sel?PALE:i%2?WHITE:LGRAY,color:TEXT}),
          ]}))
        ]
      }),

      sp(240),
      heading("3. Holdout ve Group CV Sonuçları"),
      para("Aşağıdaki tabloda her veri adayı ve model kombinasyonu için Group K-Fold CV sonuçları yer almaktadır. "+
           "Koyu vurgulanan satır, final benchmark olarak seçilen konfigürasyonu göstermektedir."),

      new Table({
        width:{size:9840,type:WidthType.DXA},columnWidths:[2100,1680,1440,960,960,960,1740],
        rows:[
          new TableRow({tableHeader:true,children:[
            hCell("Veri",2100),hCell("Model",1680),hCell("Feature",1440),
            hCell("Accuracy",960),hCell("F1",960),hCell("ROC-AUC",960),hCell("Min Ana",1740)]}),
          ...([
            ["Orijinal PIMA","Extra Trees","Insulin Çıkarıldı","0.759 ±0.033","0.657 ±0.074","0.824 ±0.043","0.614 ±0.068",false],
            ["Orijinal PIMA","XGBoost","Insulin+Skin Çıkarıldı","0.765 ±0.021","0.677 ±0.068","0.835 ±0.036","0.589 ±0.054",false],
            ["PIMA+2500/sınıf","Random Forest","Tüm Feature","0.794 ±0.032","0.792 ±0.025","0.882 ±0.026","0.711 ±0.025",false],
            ["PIMA+2500/sınıf","Random Forest","Insulin Çıkarıldı","0.787 ±0.030","0.779 ±0.041","0.873 ±0.028","0.701 ±0.043",false],
            ["PIMA+2700/sınıf","XGBoost","Skin Thickness Çıkarıldı","0.805 ±0.032","0.808 ±0.031","0.890 ±0.037","0.782 ±0.054",true],
            ["PIMA+2700/sınıf","XGBoost","Tüm Feature","0.805 ±0.035","0.809 ±0.037","0.888 ±0.035","0.769 ±0.045",false],
            ["PIMA+5000/sınıf","Extra Trees","Skin Thickness Çıkarıldı","0.807 ±0.035","0.812 ±0.031","0.886 ±0.029","0.763 ±0.050",false],
            ["PIMA+5000/sınıf","XGBoost","Insulin Çıkarıldı","0.793 ±0.019","0.798 ±0.018","0.884 ±0.019","0.741 ±0.047",false],
          ]).map(([v,m,f,a,f1,r,mn,sel],i)=>new TableRow({children:[
            cell(v, {w:2100,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,size:18}),
            cell(m, {w:1680,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,size:18}),
            cell(f, {w:1440,fill:sel?PALE:i%2?WHITE:LGRAY,color:TEXT,size:17}),
            cell(a, {w:960,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER,size:18}),
            cell(f1,{w:960,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER,size:18}),
            cell(r, {w:960,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER,size:18}),
            cell(mn,{w:1740,fill:sel?MID:i%2?WHITE:LGRAY,bold:sel,color:sel?WHITE:TEXT,align:AlignmentType.CENTER,size:18}),
          ]}))
        ]
      }),
      sp(160),
      img(chartPath("r2_model_comparison.png"),656,310),

      sp(260),
      heading("4. Final Model Sonuçları"),
      summaryTable([
        ["Final Veri Adayı","PIMA + Sentetik 2700/sınıf"],
        ["Model","XGBoost"],
        ["Çıkarılan Feature","SkinThickness"],
        ["Threshold","0.535"],
        ["Holdout Accuracy","%81.25"],
        ["Holdout F1","%81.76"],
        ["Holdout ROC-AUC","%89.14"],
        ["Group CV Accuracy","%80.46 ± 0.032"],
      ]),

      sp(240),
      heading("5. External Holdout ve Dış Kontrol Sonuçları"),
      para("Orijinal external holdout, sentetik üretimden önce ayrılmış ve model geliştirme sürecine "+
           "dahil edilmemiştir. Bu nedenle sonuç, ayrı bir dış kontrol performansı olarak değerlendirilmiştir."),

      new Table({
        width:{size:9840,type:WidthType.DXA},columnWidths:[3600,3120,3120],
        rows:[
          new TableRow({tableHeader:true,children:[
            hCell("Metrik",3600),hCell("Sentetik Holdout",3120),hCell("External Holdout (Gerçek PIMA)",3120)]}),
          ...([
            ["Accuracy","%81.25","%75.97"],
            ["Precision","%80.46","%63.49"],
            ["Recall / Sensitivity","%81.76","%74.07"],
            ["Specificity","%80.00","%77.00"],
            ["F1","%81.76","%68.38"],
            ["ROC-AUC","%89.14","%81.85"],
            ["Balanced Accuracy","—","%75.54"],
            ["MCC","—","0.496"],
            ["Brier","—","0.178"],
          ]).map(([m,s,e],i)=>new TableRow({children:[
            cell(m,{w:3600,fill:i%2?WHITE:LGRAY,bold:true,color:DARK}),
            cell(s,{w:3120,fill:i%2?WHITE:LGRAY,bold:true,color:MID,align:AlignmentType.CENTER}),
            cell(e,{w:3120,fill:i%2?WHITE:LGRAY,color:TEXT,align:AlignmentType.CENTER}),
          ]}))
        ]
      }),
      sp(160),
      img(chartPath("r2_final_metrics.png"),656,280),
      sp(80),
    ]
  }]
});
const buf=await Packer.toBuffer(doc);
fs.writeFileSync(reportPath("02_source_id_benchmark_report.docx"),buf);
console.log("Rapor 2 tamamlandı.");
}

// ═══════════════════════════════════════════════════════════════════════════
//  RAPOR 3
// ═══════════════════════════════════════════════════════════════════════════
async function buildR3(){
const doc=new Document({
  styles:{default:{document:{run:{font:"Arial",size:19,color:TEXT}}}},
  sections:[{
    properties:pageProps(),
    headers:{default:makeHeader("PIMA Source ID Kontrollü Sentetik Benchmark Raporu","Soft Voting Ensemble · 2025")},
    children:[
      coverBlock(
        "PIMA Diyabet Veri Seti İçin Source ID Kontrollü Sentetik Benchmark Raporu",
        "Soft Voting Ensemble · Leakage Kontrolü · Min Ana Metrik ≥ 0.90",
        "PIMA Indians Diabetes"
      ),
      sp(200),

      heading("Çalışma Özeti ve Hedef"),
      noteBox(
        "Ana hedef ham PIMA'da klinik genellenebilirlik kanıtlamak değil; kaynak aile kimliği (source_id) "+
        "ayrımıyla kontrol edilmiş sentetik benchmark üzerinde minimum ana metrik değerini 0.90 üzerinde "+
        "tutan bir makine öğrenmesi akışı oluşturmaktır."
      ),
      sp(160),
      summaryTable([
        ["Orijinal PIMA","768 satır, 500 negatif / 268 pozitif"],
        ["Geliştirme Verisi","614 satır"],
        ["Orijinal Dış Kontrol","154 satır"],
        ["Ana Hedef","Sentetik benchmarkta min ana metrik ≥ 0.90"],
        ["Final Aday","Kontrollü sentetik PIMA benchmarkı (2500/2500)"],
        ["Final Model","XGBoost + LightGBM + ExtraTrees soft voting"],
        ["Karar Eşiği","0.52"],
        ["Sentetik Holdout Min Ana","%96.67"],
        ["Sentetik CV Min Ana","%96.20 ± 0.67"],
        ["Sızıntı Durumu","Temiz"],
      ]),

      sp(240),
      heading("1. Leakage Kontrolü ve Kaynak Aile Yapısı"),

      leakageTable([
        ["Eğitim/test kaynak aile kesişimi","0","0","Sızıntı yok"],
        ["CV katmanı kaynak aile kesişimi","0","0","Fold izolasyonu sağlandı"],
        ["Birebir kopya sayısı","0","0","Kopya üretimi yapılmadı"],
        ["Çok yakın benzerlik oranı","0'a yakın","0.0000","Aşırı benzerlik yok"],
        ["Minimum mesafe","Raporlanır","0.1672","Yeterli ayrışma"],
        ["Orijinal dış kontrol izolasyonu","Evet","✓","Üretimden önce ayrıldı"],
        ["Bağımsız sentetik kaynak aile","Hayır","✓","Tüm örnekler dev ailesine bağlı"],
      ]),

      sp(240),
      heading("2. Sentetik Veri Üretimi ve Aday Boyutları"),

      new Table({
        width:{size:9840,type:WidthType.DXA},columnWidths:[1560,1680,1800,1680,3120],
        rows:[
          new TableRow({tableHeader:true,children:[
            hCell("Sınıf Başı",1560),hCell("Orijinal Dev",1680),hCell("Sentetik",1800),
            hCell("Toplam",1680),hCell("Negatif / Pozitif",3120)]}),
          ...([
            ["2500","614","4.386","5.000","2.500 / 2.500",true],
            ["2700","614","4.786","5.400","2.700 / 2.700",false],
            ["3000","614","5.386","6.000","3.000 / 3.000",false],
            ["4000","614","7.386","8.000","4.000 / 4.000",false],
            ["5000","614","9.386","10.000","5.000 / 5.000",false],
            ["7500","614","14.386","15.000","7.500 / 7.500",false],
            ["10000","614","19.386","20.000","10.000 / 10.000",false],
          ]).map(([sb,od,sy,t,d,sel],i)=>new TableRow({children:[
            cell(sb,{w:1560,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER}),
            cell(od,{w:1680,fill:sel?PALE:i%2?WHITE:LGRAY,align:AlignmentType.CENTER}),
            cell(sy,{w:1800,fill:sel?PALE:i%2?WHITE:LGRAY,align:AlignmentType.CENTER}),
            cell(t, {w:1680,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER}),
            cell(d, {w:3120,fill:sel?PALE:i%2?WHITE:LGRAY,color:TEXT}),
          ]}))
        ]
      }),
      sp(160),
      img(chartPath("r3_boyut_metrik.png"),656,268),

      sp(260),
      heading("3. Final Model Sonuçları"),
      summaryTable([
        ["Final Aday","Kontrollü sentetik PIMA benchmarkı (2500/2500)"],
        ["Model","XGBoost + LightGBM + ExtraTrees soft voting"],
        ["Değişken Seti","Tüm orijinal PIMA değişkenleri"],
        ["Ön İşleme","Ağaç tabanlı modeller için ek ölçekleme uygulanmadı"],
        ["Karar Eşiği","0.52"],
      ]),
      sp(160),

      new Table({
        width:{size:9840,type:WidthType.DXA},columnWidths:[3120,3360,3360],
        rows:[
          new TableRow({tableHeader:true,children:[
            hCell("Metrik",3120),hCell("Sentetik Holdout",3360),hCell("Group CV Ort. ± Std",3360)]}),
          ...([
            ["Accuracy","%96.77","%96.90 ± 0.60"],
            ["Precision","%96.86","%96.45 ± 1.03"],
            ["Recall / Sensitivity","%96.67","%97.40 ± 0.60"],
            ["Specificity","%96.87","%96.40 ± 1.08"],
            ["F1","%96.77","%96.92 ± 0.59"],
            ["ROC-AUC","%99.62","%99.67 ± 0.17"],
            ["Min Ana Metrik","%96.67","%96.20 ± 0.67"],
          ]).map(([m,h,c],i)=>new TableRow({children:[
            cell(m,{w:3120,fill:i%2?WHITE:LGRAY,bold:true,color:DARK}),
            cell(h,{w:3360,fill:i%2?WHITE:LGRAY,bold:true,color:MID,align:AlignmentType.CENTER}),
            cell(c,{w:3360,fill:i%2?WHITE:LGRAY,color:TEXT,align:AlignmentType.CENTER}),
          ]}))
        ]
      }),
      sp(160),
      img(chartPath("r3_confusion.png"),656,262),

      sp(260),
      heading("4. Sentetik Benchmark Sonucunun Yorumlanması"),
      para("Çapraz model analizi, performans artışının ağırlıklı olarak model değişiminden değil "+
           "sentetik veri üretim stratejisinden kaynaklandığını göstermiştir."),

      new Table({
        width:{size:9840,type:WidthType.DXA},columnWidths:[3000,1200,1200,1200,840,1200,1200],
        rows:[
          new TableRow({tableHeader:true,children:[
            hCell("Deney",3000),hCell("Min Ana",1200),hCell("Accuracy",1200),
            hCell("ROC-AUC",1200),hCell("Eşik",840),hCell("Shift",1200),hCell("Durum",1200)]}),
          ...([
            ["Önceki üretim + XGBoost (Skin Çıkarılmış)","0.7985","0.8161","0.8914","0.57","0.084","Doğal ref.",false],
            ["Yeni üretim + XGBoost (Skin Çıkarılmış)","0.9491","0.9618","0.9945","0.66","0.556","Gelişti",false],
            ["Önceki üretim + soft voting ensemble","0.7804","0.8161","0.8969","0.54","0.084","Model farkı yok",false],
            ["Yeni üretim + soft voting ensemble","0.9667","0.9677","0.9962","0.52","0.556","Final",true],
          ]).map(([d,mn,a,r,e,sh,s,sel],i)=>new TableRow({children:[
            cell(d, {w:3000,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,size:18}),
            cell(mn,{w:1200,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER}),
            cell(a, {w:1200,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER}),
            cell(r, {w:1200,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER}),
            cell(e, {w:840, fill:sel?PALE:i%2?WHITE:LGRAY,align:AlignmentType.CENTER}),
            cell(sh,{w:1200,fill:sel?PALE:i%2?WHITE:LGRAY,align:AlignmentType.CENTER}),
            cell(s, {w:1200,fill:sel?MID:i%2?WHITE:LGRAY,bold:sel,color:sel?WHITE:TEXT,align:AlignmentType.CENTER}),
          ]}))
        ]
      }),
      sp(160),
      img(chartPath("r3_capraz_model.png"),656,295),

      sp(260),
      heading("5. Dış Kontrol, Literatür ve Sınırlılıklar"),
      noteBox(
        "Orijinal dış kontrol performansının sentetik benchmarka göre daha düşük kalması, "+
        "modelin gerçek PIMA dağılımına transferinde sınırlılık bulunduğunu göstermektedir. "+
        "Bu sonuç final sentetik benchmark başarısını geçersiz kılmaz; ancak çalışmanın klinik "+
        "genellenebilirlik iddiası taşımadığını açık biçimde ortaya koyar."
      ),
      sp(160),
      summaryTable([
        ["Dış Kontrol Accuracy","%73.38"],
        ["Dış Kontrol F1","%60.19"],
        ["Dış Kontrol ROC-AUC","%81.67"],
        ["Dış Kontrol Min Ana Metrik","%57.41"],
        ["Brier","0.183"],
      ]),
      sp(80),
    ]
  }]
});
const buf=await Packer.toBuffer(doc);
fs.writeFileSync(reportPath("03_controlled_synthetic_benchmark_report.docx"),buf);
console.log("Rapor 3 tamamlandı.");
}

// ═══════════════════════════════════════════════════════════════════════════
//  RAPOR 4
// ═══════════════════════════════════════════════════════════════════════════
async function buildR4(){
const doc=new Document({
  styles:{default:{document:{run:{font:"Arial",size:19,color:TEXT}}}},
  sections:[{
    properties:pageProps(),
    headers:{default:makeHeader("PIMA Küçükten Büyük Ölçeğe Sentetik Benchmark Raporu","Min Savunulabilir Boyut · 2025")},
    children:[
      coverBlock(
        "PIMA Veri Setinde Kaynak Aile Kontrollü Küçükten Büyük Ölçeğe Sentetik Benchmark Raporu",
        "Minimum Savunulabilir Benchmark Boyutu · ExtraTrees · Literatür Protokol Karşılaştırması",
        "PIMA Indians Diabetes"
      ),
      sp(200),

      heading("Çalışma Özeti"),
      noteBox(
        "Bu raporda amaç, mevcut 2500/2500 referans sonucunu bozmadan daha küçük ve savunulabilir "+
        "bir PIMA + sentetik benchmark boyutu aramaktır. Başarı ölçütü: sentetik benchmark holdout "+
        "tarafında minimum ana metrik ≥ %92 ve doğruluk ≥ %93."
      ),
      sp(160),
      summaryTable([
        ["Orijinal PIMA","768 satır"],
        ["Geliştirme Verisi","614 satır"],
        ["Dış Kontrol Verisi","154 satır"],
        ["Başarı Hedefi","Holdout min ana metrik ≥ %92 ve doğruluk ≥ %93"],
        ["Seçilen Aday","1000/1000 — yakınlaştırma 0.60, adaptif gürültü profili"],
        ["Model","ExtraTrees"],
        ["Holdout Doğruluk","%94.16 (→ %93.93 düzeltilmiş)"],
        ["Holdout Min Ana Metrik","%92.49"],
        ["Grup CV Min Ana","%91.31 ± 0.008"],
        ["Cohen's d","0.952"],
        ["Dağılım Kayması","0.100"],
        ["Sızıntı Durumu","Temiz"],
      ]),

      sp(240),
      heading("1. Doğrulama Protokolü ve Veri Sızıntısı Kontrolü"),

      leakageTable([
        ["Kaynak aile takibi (Source ID)","Her sentetik örnek kaynak aile kimliği taşıdı","✓","Aile takibi sağlandı"],
        ["Dış kontrol izolasyonu","Üretimden önce ayrıldı","✓","Sızıntı riski yok"],
        ["Grup çapraz doğrulama","StratifiedGroupKFold ile kaynak aileler bölünmedi","✓","CV temiz"],
        ["Kopya kontrolü","Birebir kopya ve yakın benzerlik tarandı","✓","Kopya yok"],
        ["Sentetik/orijinal train-test ayrımı","Train ve test kaynak aile kesişimi: 0","✓","Leakage yok"],
        ["Dış kontrol kaynak çakışması","Dış kontrol kümesinde sentetik kaynak aile: 0","✓","İzolasyon tam"],
        ["Bağımsız sentetik kaynak aile","Tüm sentetik örnekler dev ailesine bağlı","✓","Kontrol tam"],
      ]),

      sp(240),
      heading("2. Aday Taraması Sonuçları"),
      para("Her sınıf başı hedefte en iyi holdout sonucu ve grup çapraz doğrulama özeti aşağıdaki tabloda "+
           "gösterilmektedir. Seçilen adayın neden öne çıktığı ve 2500/2500 referansının neden daha büyük "+
           "kaldığı bu tablo üzerinden izlenebilir."),

      new Table({
        width:{size:9840,type:WidthType.DXA},columnWidths:[1080,1800,1080,1080,1680,840,720,1560],
        rows:[
          new TableRow({tableHeader:true,children:[
            hCell("Aday",1080),hCell("En İyi Model",1800),hCell("Doğruluk",1080),
            hCell("Min Ana",1080),hCell("Grup CV Min Ana",1680),hCell("Cohen's d",840),
            hCell("Shift",720),hCell("Durum",1560)]}),
          ...([
            ["500/500","XGB+LGBM+ET Soft Voting","%92.00","%91.00","%85.60 ±0.026","0.836","0.115","Geçmedi",false,RED],
            ["650/650","XGB+LGBM+ET Soft Voting","%91.99","%91.61","%87.38 ±0.021","1.050","0.125","Geçmedi",false,RED],
            ["800/800","XGB+LGBM+ET Soft Voting","%91.19","%91.19","%90.00 ±0.012","1.281","0.117","Geçmedi",false,RED],
            ["1000/1000","ExtraTrees","%93.93","%93.43","%91.80 ±0.012","0.952","0.100","Seçildi",true,MID],
            ["1250/1250","XGB+LGBM+ET Soft Voting","%94.16","%92.61","%92.56 ±0.018","1.026","0.093","Geçti, büyük",false,AMBER],
            ["1500/1500","XGB+LGBM+ET Soft Voting","%94.80","%94.79","%93.80 ±0.025","1.027","0.092","Geçti, büyük",false,AMBER],
            ["2000/2000","LightGBM","%96.39","%96.38","%95.54 ±0.003","1.086","0.089","Geçti, büyük",false,AMBER],
            ["2250/2250","ExtraTrees","%96.76","%96.64","%95.95 ±0.003","1.092","0.090","Geçti, büyük",false,AMBER],
            ["2500/2500","ExtraTrees","%96.99","%96.79","%96.44 ±0.004","1.124","0.086","Referans",false,MID],
          ]).map(([a,m,d,mn,cv,cd,sh,st,sel,stColor],i)=>new TableRow({children:[
            cell(a, {w:1080,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,size:18}),
            cell(m, {w:1800,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,size:17}),
            cell(d, {w:1080,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER}),
            cell(mn,{w:1080,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER}),
            cell(cv,{w:1680,fill:sel?PALE:i%2?WHITE:LGRAY,bold:sel,color:sel?DARK:TEXT,align:AlignmentType.CENTER,size:17}),
            cell(cd,{w:840, fill:sel?PALE:i%2?WHITE:LGRAY,align:AlignmentType.CENTER,size:18}),
            cell(sh,{w:720, fill:sel?PALE:i%2?WHITE:LGRAY,align:AlignmentType.CENTER,size:18}),
            cell(st,{w:1560,fill:sel?MID:"F8F9F9",bold:sel||stColor===RED,
                    color:sel?WHITE:stColor,align:AlignmentType.CENTER,size:17}),
          ]}))
        ]
      }),
      sp(160),
      img(chartPath("r4_aday_tarama.png"),656,310),
      sp(160),
      img(chartPath("r4_cohens_shift.png"),656,262),

      sp(260),
      heading("3. Seçilen Küçük Benchmark Adayı"),
      summaryTable([
        ["Veri Adayı","1000/1000; yakınlaştırma 0.60; adaptif gürültü profili"],
        ["Model","ExtraTrees"],
        ["Değişken Seti","Tüm orijinal PIMA değişkenleri"],
        ["Karar Eşiği","0.51"],
        ["Holdout Doğruluk","%94.16"],
        ["Holdout Min Ana Metrik","%92.49"],
        ["Grup CV Min Ana Metrik","%91.31 ± 0.008"],
        ["Cohen's d","0.952"],
        ["Dağılım Kayması","0.100"],
      ]),

      sp(240),
      heading("4. Literatürle Protokol Karşılaştırması"),

      new Table({
        width:{size:9840,type:WidthType.DXA},columnWidths:[2640,3600,3600],
        rows:[
          new TableRow({tableHeader:true,children:[
            hCell("Karşılaştırma Ekseni",2640),hCell("Literatürde Sık Görülen",3600),hCell("Bu Çalışmadaki Yaklaşım",3600)]}),
          ...([
            ["Veri Artırma","SMOTE, ADASYN, CTGAN, SMOTE-ENN","Kaynak aile kontrollü sentetik benchmark"],
            ["Veri Bölme","Çalışmadan çalışmaya değişken","Dış kontrol sentetik üretimden önce ayrıldı"],
            ["Kaynak Aile Takibi","Genellikle raporlanmaz","Her örnek source_id kimliği taşıdı"],
            ["Sızıntı Kontrolü","Her çalışmada aynı detayda değil","Kaynak aile, kopya, yakın benzerlik, dış kontrol"],
            ["Çapraz Doğrulama","k-fold / stratified k-fold","StratifiedGroupKFold (source_id bazlı)"],
            ["Sonuç Yorumu","Genel başarı gibi sunulabilir","Sentetik benchmark ve dış kontrol ayrı yorumlandı"],
          ]).map(([k,l,b],i)=>new TableRow({children:[
            cell(k,{w:2640,fill:i%2?WHITE:LGRAY,bold:true,color:DARK}),
            cell(l,{w:3600,fill:i%2?WHITE:LGRAY,color:TEXT,size:18}),
            cell(b,{w:3600,fill:i%2?PALE:"F0FFF4",bold:true,color:MID,size:18}),
          ]}))
        ]
      }),

      sp(240),
      heading("5. Dış Kontrol Sonuçları ve Sınırlılıklar"),
      noteBox(
        "Doğrulama protokolü ve veri sızıntısı kontrol seviyesi çalışmadan çalışmaya değiştiği için "+
        "skorlar doğrudan kıyaslanmamalıdır. Bu rapordaki yüksek metrikler kontrollü sentetik "+
        "benchmark bağlamında değerlendirilmelidir."
      ),
      sp(160),
      new Table({
        width:{size:9840,type:WidthType.DXA},columnWidths:[3000,2280,2280,2280],
        rows:[
          new TableRow({tableHeader:true,children:[
            hCell("Dış Kontrol Metriği",3000),hCell("Rapor 2 (2700/sınıf)",2280),
            hCell("Rapor 3 (2500/sınıf)",2280),hCell("Rapor 4 (1000/sınıf)",2280)]}),
          ...([
            ["Accuracy (Doğruluk)","%75.97","%73.38","%74.03"],
            ["F1","%68.38","%60.19","%62.26"],
            ["ROC-AUC","%81.85","%81.67","%81.81"],
            ["Min Ana Metrik","—","%57.41","%61.11"],
            ["Brier","0.178","0.183","0.169"],
          ]).map(([m,r2,r3,r4],i)=>new TableRow({children:[
            cell(m, {w:3000,fill:i%2?WHITE:LGRAY,bold:true,color:DARK}),
            cell(r2,{w:2280,fill:i%2?WHITE:LGRAY,color:TEXT,align:AlignmentType.CENTER}),
            cell(r3,{w:2280,fill:i%2?WHITE:LGRAY,color:TEXT,align:AlignmentType.CENTER}),
            cell(r4,{w:2280,fill:i%2?WHITE:LGRAY,bold:true,color:MID,align:AlignmentType.CENTER}),
          ]}))
        ]
      }),
      sp(160),
      img(chartPath("r4_dis_kontrol_karsilastirma.png"),656,268),
      sp(80),
    ]
  }]
});
const buf=await Packer.toBuffer(doc);
fs.writeFileSync(reportPath("04_scale_sweep_benchmark_report.docx"),buf);
console.log("Rapor 4 tamamlandı.");
}

// ─── Hepsini çalıştır ────────────────────────────────────────────────────
(async()=>{
  await buildR1();
  await buildR2();
  await buildR3();
  await buildR4();
  console.log("Tüm raporlar hazır.");
})();
