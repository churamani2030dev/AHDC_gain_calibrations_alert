// gain_MPV_calib_check_final.groovy
//
// Author: Churamani Paudel
//
// Two–mode per–wire ADC gain calibration for AHDC:
//   MODE = "calib":
//     * Build per–wire SIG/BKG histograms for elastic events
//       (W² window, Δφ≈180°, vz, FD e−, KF track quality, track association).
//     * Species gate:
//         - "deut" : ΣADC–pT banana (deuteron band)
//         - "prot" : ΣADC window with banana veto
//     * α from control window (pedestal region): robust S/B → clamp [ALPHA_MIN, ALPHA_MAX]
//     * SUB = SIG − α·BKG (optionally non–negative NNLS–like).
//     * Global Landau on SUM_SUB → MPV_REF & error.
//     * Per–wire Landau fits on SUB → MPV_pre & error.
//     * Gain per wire: gain = MPV_REF / MPV_pre; error propagated from MPV errors.
//     * Build gain–corrected SUB per wire; optional second Landau fit → MPV_post.
//     * Write CSV with α, MPVs, χ²/ndf, gains, errors.
//     * GUI#1: per–wire SUB / SUBgain, SUM_SIG/BKG/SUB, MPV_pre/post vs wire, gain vs wire.
//     * GUI#2: per–wire overlay (SUB vs SUBgain), SIG vs α·BKG, pT–ADC pre/post (and L/R).
//
//   MODE = "check":
//     * Read gains (and MPV_pre/post) from CSV.
//     * Apply gains to new data (same selections & species).
//     * Refill SUB, SUB gain–corrected, pT–ADC 2D, graphs from CSV values.
//     * All QA plots use CSV gains; no refitting required.
//
// Typical usage:
//   run-groovy gain_MPV_calib_check_final.groovy process \
//       -mode calib -species deut -mode_valid banana \
//       -trackid 1 -nevent 5000000000 \
//       -csv gains_calib_deut.csv \
//       files.hipo ...
//
//   run-groovy gain_MPV_calib_check_final.groovy process \
//       -mode check -species deut -mode_valid banana \
//       -trackid 1 -csv gains_calib_deut.csv \
//       files.hipo ...

import java.util.*
import javax.swing.JFrame
import java.io.File

import org.jlab.io.hipo.HipoDataSource
import org.jlab.io.base.DataEvent
import org.jlab.io.base.DataBank

import org.jlab.groot.data.H1F
import org.jlab.groot.data.H2F
import org.jlab.groot.data.GraphErrors
import org.jlab.groot.graphics.EmbeddedCanvasTabbed
import org.jlab.groot.math.F1D
import org.jlab.groot.fitter.DataFitter
import org.jlab.groot.math.FunctionFactory

import org.jlab.jnp.utils.options.OptionStore
import org.jlab.jnp.utils.benchmark.ProgressPrintout
import groovy.transform.Field

// ------------------------------ GLOBAL CONFIG ------------------------------
@Field double EBEAM       = 2.24d
@Field double M_D         = 1.875612d

// elastic W² window (D target)
@Field double W2_MIN      = 3.46d
@Field double W2_MAX      = 3.67d

// Δφ ~ 180°
@Field double DPHI_HALF   = 10.0d

// electron vertex & FD
@Field double VZ_MIN      = -20.0d
@Field double VZ_MAX      = +10.0d
@Field boolean FD_ONLY    = true

// KF track quality
@Field int    KF_NHITS_MIN= 8
@Field double KF_CHI2_MAX = 30.0d

// pT slice for calibration
@Field double PT_SLICE_MIN= 0.24d
@Field double PT_SLICE_MAX= 0.29d

// event limit
@Field long   MAXEV       = -1L

// ADC binning
@Field int    ADC_NBINS   = 220
@Field double ADC_MAX     = 4500.0d

// control window for α (pedestal–dominated region)
@Field double CTRL_LO     = 80.0d
@Field double CTRL_HI     = 380.0d
@Field double ALPHA_MIN   = 0.001d
@Field double ALPHA_MAX   = 5.0d

// banana: ΣADC/BAN_SCALE vs pT for deuteron
@Field boolean BANANA_ON  = true
@Field double BAN_SCALE   = 1000.0d
@Field double BAN_C0      = 6.0d
@Field double BAN_C1      = -35.0d
@Field double BAN_HALF    = 1.5d

// proton ΣADC window (ΣADC/BAN_SCALE), with banana veto
@Field double PROT_Y_MIN  = 2.0d
@Field double PROT_Y_MAX  = 5.5d

// Species: "deut" or "prot"
@Field String SPECIES     = "deut"

// fit window for Landau on SUB
@Field double FIT_LO      = 450.0d
@Field double FIT_HI      = 2000.0d

// SUB mode: "nnls" (non–negative) or "raw"
@Field String SUBMODE     = "nnls"

// skip drawing wires where pre–fit failed in some canvases
@Field boolean SKIP_FAILED_PRE = false

// 2D pT–ADC QA
@Field int    PT2_NBINS   = 80
@Field double PT2_MIN     = 0.20d
@Field double PT2_MAX     = 0.45d
@Field int    ADC2_NBINS  = 220
@Field double ADC2_MIN    = 0.0d
@Field double ADC2_MAX    = 1400.0d

// MODE = "calib" or "check"
@Field String MODE        = "calib"

// CSV path
@Field String CSV_OUT     = null

// drawing layout
@Field int DRAW_COLS = 5
@Field int DRAW_ROWS = 2
int PAGE_SIZE(){ return DRAW_COLS*DRAW_ROWS }

@Field boolean HAS_ORDER = false

// ------------------------------ SMALL HELPERS ------------------------------
static double deg0to360(double a){
  double x = a % 360.0d
  return (x < 0.0d) ? x + 360.0d : x
}
static double phiDeg(float px,float py){
  return deg0to360(Math.toDegrees(Math.atan2((double)py,(double)px)))
}
static double dphi0to360(double pe,double pk){
  double x = (pe - pk) % 360.0d
  return (x < 0.0d) ? x + 360.0d : x
}
static boolean between(double x,double lo,double hi){
  return (x >= lo && x <= hi)
}
static int Lenc(int sl,int l){ return 10*sl + l }
static double clampADC(double A,double max){
  return Math.min(max - 1e-6, A)
}
static double parseDoubleSafe(String s){
  if(s==null) return Double.NaN
  try{ return Double.parseDouble(s.trim()) }catch(Exception e){ return Double.NaN }
}

// banana in (pT, ΣADC_track)
boolean inBanana(double pt,long sumADC_trk){
  double y = ((double)sumADC_trk)/BAN_SCALE
  double c = BAN_C0 + BAN_C1*(pt - 0.26d)
  return Math.abs(y - c) <= BAN_HALF
}

// proton window in ΣADC with banana veto
boolean inProton(double pt,long sumADC_trk){
  double y = ((double)sumADC_trk)/BAN_SCALE
  if(y<PROT_Y_MIN || y>PROT_Y_MAX) return false
  if(inBanana(pt,sumADC_trk)) return false
  return true
}

double clampAlpha(double a){
  if(Double.isNaN(a) || Double.isInfinite(a)) return ALPHA_MIN
  return Math.max(ALPHA_MIN,Math.min(ALPHA_MAX,a))
}

// symmetric Y-range around zero
class YSym{
  double ymin,ymax
  YSym(double a,double b){ ymin=a; ymax=b }
}
YSym symmetricY(H1F h,double pad){
  int n = h.getAxis().getNBins()
  double ymin=0.0,ymax=0.0
  for(int b=0;b<n;b++){
    double y = h.getBinContent(b)
    if(b==0){ ymin=y; ymax=y }
    else{
      if(y<ymin) ymin=y
      if(y>ymax) ymax=y
    }
  }
  double a = Math.max(Math.abs(ymin),Math.abs(ymax))
  a = (a<=0 ? 1.0 : a*pad)
  return new YSym(-a,+a)
}

void setPadRangesSafe(def canvas,int padIndex,double xmin,double xmax,double ymin,double ymax){
  try{
    canvas.getPad(padIndex).setAxisRange(xmin,xmax,ymin,ymax)
    return
  }catch(Exception e){}
  try{
    def pad = canvas.getPad(padIndex)
    def af  = pad.getAxisFrame()
    try{ af.getAxisX().setRange(xmin,xmax) }catch(Exception e1){}
    try{ af.getAxisY().setRange(ymin,ymax) }catch(Exception e2){}
  }catch(Exception ee){}
}

// build gain–corrected SUB
H1F buildGainCorrHist(H1F hSub, def wk, double gain){
  int nb  = hSub.getAxis().getNBins()
  double xlo=hSub.getAxis().min()
  double xhi=hSub.getAxis().max()
  if(nb<=0){ nb=ADC_NBINS; xlo=0.0; xhi=ADC_MAX }

  H1F hCorr = new H1F("sub_gain_"+wk.toString().replace(' ','_'),
                      "ADC_{corr} — "+wk.toString()+";ADC_{corr};Counts",
                      nb,xlo,xhi)
  double gUsed=(gain>0.0 && !Double.isNaN(gain)) ? gain : 1.0

  def axOld  = hSub.getAxis()
  int nOld   = axOld.getNBins()
  double xMinOld = axOld.min()
  double xMaxOld = axOld.max()
  double bwOld   = (xMaxOld-xMinOld)/Math.max(1,nOld)

  def axNew  = hCorr.getAxis()
  int nNew   = axNew.getNBins()
  double xMinNew = axNew.min()
  double xMaxNew = axNew.max()
  double bwNew   = (xMaxNew-xMinNew)/Math.max(1,nNew)

  for(int b=0;b<nOld;b++){
    double c = hSub.getBinContent(b)
    if(c==0) continue
    double x = axOld.getBinCenter(b)
    double xCorr = clampADC(x*gUsed,ADC_MAX)
    if(xCorr<xMinNew || xCorr>=xMaxNew) continue
    int ibNew = (int)Math.floor((xCorr - xMinNew)/bwNew)
    if(ibNew>=0 && ibNew<nNew){
      double oldVal = hCorr.getBinContent(ibNew)
      hCorr.setBinContent(ibNew,oldVal+c)
    }
  }
  hCorr.setLineColor(6) // magenta
  return hCorr
}

// ------------------------------ DATA CLASSES ------------------------------
class RecP{
  int pid
  float px,py,pz
  float vx,vy,vz,vt
  byte charge
  float beta,chi2pid
  short status
}

class KFRec{
  int   idx   = -1
  float px,py,pz,chi2
  int   nhits
  double phiDeg = Double.NaN
  int   trackid = -1
}

final class SLW{
  final int sl,l,w
  SLW(int a,int b,int c){ sl=a; l=b; w=c }
  int hashCode(){ ((sl*1315423911)^(l*2654435761))^w }
  boolean equals(Object o){
    if(!(o instanceof SLW)) return false
    SLW x=(SLW)o
    return x.sl==sl && x.l==l && x.w==w
  }
}

final class SL{
  final int sl,l
  SL(int a,int b){ sl=a; l=b }
  int hashCode(){ (sl*1315423911)^l }
  boolean equals(Object o){
    if(!(o instanceof SL)) return false
    SL x=(SL)o
    return x.sl==sl && x.l==l
  }
}

final class WireKey{
  final int s,Lraw,c
  WireKey(int s,int Lraw,int c){
    this.s=s; this.Lraw=Lraw; this.c=c
  }
  int hashCode(){ (s*73856093)^(Lraw*19349663)^(c*83492791) }
  boolean equals(Object o){
    if(!(o instanceof WireKey)) return false
    WireKey k=(WireKey)o
    return (k.s==s && k.Lraw==Lraw && k.c==c)
  }
  String toString(){ String.format("S%d L%02d C%d",s,Lraw,c) }
}

// ------------------------------ BANK READERS ------------------------------
RecP getElectronREC(DataEvent ev){
  if(!ev.hasBank("REC::Particle")) return null
  DataBank b = ev.getBank("REC::Particle")
  int best=-1
  for(int i=0;i<b.rows();i++){
    if(b.getInt("pid",i)!=11) continue
    float vz = b.getFloat("vz",i)
    short st = b.getShort("status",i)
    if(vz<VZ_MIN || vz>VZ_MAX) continue
    if(FD_ONLY && st>=0) continue
    best=i
    break
  }
  if(best<0) return null
  RecP e=new RecP()
  e.pid    = b.getInt("pid",best)
  e.px     = b.getFloat("px",best)
  e.py     = b.getFloat("py",best)
  e.pz     = b.getFloat("pz",best)
  e.vx     = b.getFloat("vx",best)
  e.vy     = b.getFloat("vy",best)
  e.vz     = b.getFloat("vz",best)
  e.vt     = b.getFloat("vt",best)
  e.charge = b.getByte("charge",best)
  e.beta   = b.getFloat("beta",best)
  e.chi2pid= b.getFloat("chi2pid",best)
  e.status = b.getShort("status",best)
  return e
}

double W_from_e(double Ebeam,RecP e){
  double Ee = Math.sqrt((double)e.px*e.px + (double)e.py*e.py + (double)e.pz*e.pz)
  double qx = -(double)e.px
  double qy = -(double)e.py
  double qz = Ebeam - (double)e.pz
  double q0 = Ebeam - Ee
  double Eh = M_D + q0
  double w2 = Eh*Eh - (qx*qx+qy*qy+qz*qz)
  return (w2>0.0d)? Math.sqrt(w2) : Double.NaN
}

// best KF track back-to-back with electron
KFRec bestKF_BackToBack(DataEvent ev,double phi_e){
  KFRec out=new KFRec()
  if(!ev.hasBank("AHDC::kftrack")) return out
  DataBank k = ev.getBank("AHDC::kftrack")
  double bestAbs=Double.POSITIVE_INFINITY
  for(int i=0;i<k.rows();i++){
    int nh=k.getInt("n_hits",i)
    if(nh<KF_NHITS_MIN) continue
    float chi2=k.getFloat("chi2",i)
    if(!Float.isNaN(chi2) && (double)chi2>KF_CHI2_MAX) continue
    float px = k.getFloat("px",i)
    float py = k.getFloat("py",i)
    double pk = phiDeg(px,py)
    double dphi = Math.abs(dphi0to360(phi_e,pk)-180.0d)
    if(dphi<bestAbs){
      bestAbs=dphi
      out.idx=i
      out.px=px
      out.py=py
      out.pz=k.getFloat("pz",i)
      out.chi2=chi2
      out.nhits=nh
      out.phiDeg=pk
      out.trackid=k.getInt("trackid",i)
    }
  }
  return out
}

// "heavy" track (largest sum_adc / n_hits)
KFRec bestKF_Heavy(DataEvent ev){
  KFRec out=new KFRec()
  if(!ev.hasBank("AHDC::kftrack")) return out
  DataBank k = ev.getBank("AHDC::kftrack")
  double best=-1.0
  for(int i=0;i<k.rows();i++){
    int nh=k.getInt("n_hits",i)
    if(nh<KF_NHITS_MIN) continue
    float chi2=k.getFloat("chi2",i)
    if(!Float.isNaN(chi2) && chi2>KF_CHI2_MAX) continue
    int sadc=0
    try{ sadc=k.getInt("sum_adc",i) }catch(Exception ignore){}
    double score = (nh>0 ? (double)sadc/nh : -1.0)
    if(score>best){
      best=score
      out.idx=i
      out.nhits=nh
      out.chi2=chi2
      out.px=k.getFloat("px",i)
      out.py=k.getFloat("py",i)
      out.pz=k.getFloat("pz",i)
      out.trackid=k.getInt("trackid",i)
      out.phiDeg=phiDeg(out.px,out.py)
    }
  }
  return out
}

// waveform gate (currently permissive; kept for future tightening)
Set<String> wfExplicitGood(DataEvent ev){
  HashSet<String> good=new HashSet<String>()
  if(!ev.hasBank("AHDC::wf")) return good
  DataBank w=ev.getBank("AHDC::wf")
  for(int i=0;i<w.rows();i++){
    int flag
    try{ flag=w.getInt("flag",i) }catch(Exception e){ continue }
    if(flag!=0 && flag!=1) continue
    int s,LencVal,c
    try{
      int sl=w.getInt("superlayer",i)
      int l =w.getInt("layer",i)
      s    =w.getInt("sector",i)
      c    =w.getInt("component",i)
      LencVal=Lenc(sl,l)
    }catch(Exception e){
      try{
        s=w.getInt("sector",i)
        c=w.getInt("component",i)
        int Lraw=w.getInt("layer",i)
        LencVal=Lraw
      }catch(Exception ee){ continue }
    }
    good.add(s+"#"+LencVal+"#"+c)
  }
  return good
}

boolean wfPassForAdcRow(DataBank a,int i,Set<String> wfGoodSet){
  int s=a.getInt("sector",i)
  int Lraw=a.getInt("layer",i)
  int c=a.getInt("component",i)
  if(wfGoodSet.contains(s+"#"+Lraw+"#"+c)) return true
  // for now: keep all; WF gate is not a hard veto
  return true
}

// track association sets
class AssocSets{
  Set<SLW> slw=new HashSet<SLW>()
  Set<SL>  sl =new HashSet<SL>()
}
AssocSets buildAssocSetsForTrackId(DataEvent ev,int wantedId){
  AssocSets as=new AssocSets()
  if(!ev.hasBank("AHDC::hits")) return as
  DataBank h=ev.getBank("AHDC::hits")
  for(int i=0;i<h.rows();i++){
    int tid=h.getInt("trackid",i)
    if(tid!=wantedId) continue
    int sl=(h.getByte("superlayer",i)&0xFF)
    int l =(h.getByte("layer",i)&0xFF)
    int w =h.getInt("wire",i)
    as.slw.add(new SLW(sl,l,w))
    as.sl.add(new SL(sl,l))
  }
  return as
}

// ------------------------------ α FROM CONTROL REGION ------------------------------
class AlphaCtrlRes{
  double alphaLSQ=0.0
  double alphaRatio=0.0
  double alphaPref=0.0
  int    nBins=0
  double Ssum=0.0
  double Bsum=0.0
}

// robust α from background–dominated bins in [amin,amax]
AlphaCtrlRes alphaFromControl(H1F sig,H1F bkg,double amin,double amax){
  def ax = sig.getAxis()
  int n  = ax.getNBins()
  double SB=0.0,BB=0.0
  int used=0
  double Ssum=0.0,Bsum=0.0

  for(int b=0;b<n;b++){
    double xC = ax.getBinCenter(b)
    if(xC<amin || xC>amax) continue
    double S = sig.getBinContent(b)
    double B = bkg.getBinContent(b)
    if(B<=0.0 && S<=0.0) continue

    // skip clearly signal–dominated bins
    if(B>0.0 && S>1.3*B) continue

    double wt = 1.0/Math.max(1.0,S+B)
    SB += wt*S*B
    BB += wt*B*B
    used++
    Ssum+=S
    Bsum+=B
  }

  AlphaCtrlRes out=new AlphaCtrlRes()
  double aLSQ  = (BB>0.0 ? SB/BB : 0.0)
  double aRatio= (Bsum>0.0 ? Ssum/Bsum : 0.0)
  double aPref = 0.0

  if(aLSQ>0.0 && aRatio>0.0)      aPref=Math.min(aLSQ,aRatio)
  else if(aLSQ>0.0)               aPref=aLSQ
  else if(aRatio>0.0)             aPref=aRatio

  out.alphaLSQ=aLSQ
  out.alphaRatio=aRatio
  out.alphaPref=aPref
  out.nBins=used
  out.Ssum=Ssum
  out.Bsum=Bsum
  return out
}

// MPV error from width / sqrt(N)
double estimateMPVError(H1F h,double mpv,double xmin,double xmax){
  if(h==null || Double.isNaN(mpv) || mpv<=0.0) return 0.0
  def ax = h.getAxis()
  int n  = ax.getNBins()
  double sumW=0.0,sumVar=0.0
  for(int b=0;b<n;b++){
    double x = ax.getBinCenter(b)
    if(x<xmin || x>xmax) continue
    double c = h.getBinContent(b)
    if(c<=0.0) continue
    sumW += c
    double dx = x-mpv
    sumVar += c*dx*dx
  }
  if(sumW<=1.0) return 0.0
  double sigmaWidth = Math.sqrt(sumVar/sumW)
  return sigmaWidth/Math.sqrt(sumW)
}

// ------------------------------ LANDAU FITS ------------------------------
class FitResultGlobal{
  boolean ok=false
  double amp,mpv,sigma,mpvErr,chi2=Double.NaN
  int ndf=0
  F1D func=null
}
FitResultGlobal fitGlobalLandau(H1F h,double xmin,double xmax){
  FitResultGlobal out=new FitResultGlobal()
  if(h.integral()<200.0) return out

  int binMax = h.getMaximumBin()
  double mpv0=h.getAxis().getBinCenter(binMax)
  double amp0=Math.max(1.0,h.getBinContent(binMax))
  double sig0=Math.max(50.0,(xmax-xmin)/10.0)
  if(mpv0<xmin || mpv0>xmax) mpv0=0.5*(xmin+xmax)

  F1D f=new F1D("f_sum","[A]*landau(x,[MPV],[SIG])",xmin,xmax)
  f.setParameter(0,amp0)
  f.setParameter(1,mpv0)
  f.setParameter(2,sig0)

  try{ DataFitter.fit(f,h,"Q") }catch(Exception e){ return out }

  out.amp   = f.getParameter(0)
  out.mpv   = f.getParameter(1)
  out.sigma = f.getParameter(2)
  out.mpvErr= estimateMPVError(h,out.mpv,xmin,xmax)

  def ax=h.getAxis()
  int nBins=ax.getNBins()
  int used=0
  double chisq=0.0
  for(int b=0;b<nBins;b++){
    double x=ax.getBinCenter(b)
    if(x<xmin || x>xmax) continue
    double y=h.getBinContent(b)
    double err=h.getBinError(b)
    if(err<=0.0) err=Math.sqrt(Math.max(1.0,Math.abs(y)))
    if(err<=0.0) continue
    double yfit=f.evaluate(x)
    double d=(y-yfit)/err
    chisq+=d*d
    used++
  }
  int nPars=f.getNPars()
  int ndf=used-nPars
  if(ndf>0){
    out.chi2=chisq
    out.ndf=ndf
  }
  f.setLineColor(3)
  f.setLineWidth(3)
  out.func=f
  out.ok=(out.mpv>0.0 && !Double.isNaN(out.mpv))
  return out
}

class FitResult1D{
  boolean ok=false
  double amp,mpv,sigma,ampErr,mpvErr,sigErr,chi2=Double.NaN
  int ndf=0
  F1D func=null
}
FitResult1D fitLandau1D(H1F h,double xmin,double xmax,String tag){
  FitResult1D out=new FitResult1D()
  if(h==null || h.integral()<80.0) return out

  int binMax = h.getMaximumBin()
  double mpv0=h.getAxis().getBinCenter(binMax)
  double amp0=Math.max(1.0,h.getBinContent(binMax))
  double sig0=Math.max(30.0,(xmax-xmin)/12.0)

  F1D f=new F1D("f_"+tag,"[A]*landau(x,[MPV],[SIG])",xmin,xmax)
  f.setParameter(0,amp0)
  f.setParameter(1,mpv0)
  f.setParameter(2,sig0)
  try{ DataFitter.fit(f,h,"Q") }catch(Exception e){ return out }

  out.amp   = f.getParameter(0)
  out.mpv   = f.getParameter(1)
  out.sigma = f.getParameter(2)
  out.ampErr=0.0
  out.sigErr=0.0
  out.mpvErr=estimateMPVError(h,out.mpv,xmin,xmax)

  def ax=h.getAxis()
  int used=0
  double chisq=0.0
  for(int b=0;b<ax.getNBins();b++){
    double x=ax.getBinCenter(b)
    if(x<xmin || x>xmax) continue
    double y=h.getBinContent(b)
    double err=h.getBinError(b)
    if(err<=0.0) err=Math.sqrt(Math.max(1.0,Math.abs(y)))
    if(err<=0.0) continue
    double yfit=f.evaluate(x)
    double d=(y-yfit)/err
    chisq+=d*d
    used++
  }
  int nPars=f.getNPars()
  int ndf=used-nPars
  if(ndf>0){
    out.chi2=chisq
    out.ndf=ndf
  }
  f.setLineColor(3)
  f.setLineWidth(3)
  out.func=f
  out.ok=(out.mpv>0.0 && !Double.isNaN(out.mpv))
  return out
}

// ------------------------------ HIST CONTAINERS ------------------------------
final class PairH{
  final H1F sig,bkg,sub
  PairH(String tag,String titleBase,int nb,double lo,double hi){
    sig=new H1F("sig_"+tag,titleBase+" (SIG);ADC;Counts",nb,lo,hi)
    bkg=new H1F("bkg_"+tag,titleBase+" (BKG);ADC;Counts",nb,lo,hi)
    sub=new H1F("sub_"+tag,titleBase+" (SUB);ADC;Counts",nb,lo,hi)
    sig.setLineColor(4)
    bkg.setLineColor(2)
    sub.setLineColor(4)
  }
}

@Field Map<WireKey,PairH>  histMap      = new LinkedHashMap<>()
@Field Map<WireKey,H1F>    subGainMap   = new LinkedHashMap<>()
@Field Map<WireKey,Double> alphaMap     = new LinkedHashMap<>()
@Field Map<WireKey,H1F>    scaledBkgMap = new LinkedHashMap<>()

@Field H1F SUM_SIG = new H1F("sum_sig","Per–wire ADC (SIG all);ADC;Counts",ADC_NBINS,0.0,ADC_MAX)
@Field H1F SUM_BKG = new H1F("sum_bkg","Per–wire ADC (BKG all);ADC;Counts",ADC_NBINS,0.0,ADC_MAX)
@Field H1F SUM_SUB = new H1F("sum_sub","Per–wire ADC (SUB all);ADC;Counts",ADC_NBINS,0.0,ADC_MAX)
void styleSum(){
  SUM_SIG.setLineColor(1)
  SUM_BKG.setLineColor(2)
  SUM_SUB.setLineColor(4)
}
styleSum()

PairH getPair(WireKey k){
  PairH p=histMap.get(k)
  if(p==null){
    String tag = k.toString().replace(' ','_')
    String ttl = "ADC — "+k.toString()
    p=new PairH(tag,ttl,ADC_NBINS,0.0,ADC_MAX)
    histMap.put(k,p)
  }
  return p
}

// 2D pT–ADC
@Field H2F H2_PRE_ALL  = new H2F("h2_pre_all","pT vs ADC (SUB pre–gain);pT (GeV);ADC",
                                 PT2_NBINS,PT2_MIN,PT2_MAX, ADC2_NBINS,ADC2_MIN,ADC2_MAX)
@Field H2F H2_POST_ALL = new H2F("h2_post_all","pT vs ADCcorr (SUB post–gain);pT (GeV);ADCcorr",
                                 PT2_NBINS,PT2_MIN,PT2_MAX, ADC2_NBINS,0.0,ADC2_MAX)
@Field H2F H2_PRE_L    = new H2F("h2_pre_L","LEFT: pT vs ADC (SUB);pT (GeV);ADC",
                                 PT2_NBINS,PT2_MIN,PT2_MAX, ADC2_NBINS,ADC2_MIN,ADC2_MAX)
@Field H2F H2_PRE_R    = new H2F("h2_pre_R","RIGHT: pT vs ADC (SUB);pT (GeV);ADC",
                                 PT2_NBINS,PT2_MIN,PT2_MAX, ADC2_NBINS,ADC2_MIN,ADC2_MAX)
@Field H2F H2_POST_L   = new H2F("h2_post_L","LEFT: pT vs ADCcorr (SUB);pT (GeV);ADCcorr",
                                 PT2_NBINS,PT2_MIN,PT2_MAX, ADC2_NBINS,0.0,ADC2_MAX)
@Field H2F H2_POST_R   = new H2F("h2_post_R","RIGHT: pT vs ADCcorr (SUB);pT (GeV);ADCcorr",
                                 PT2_NBINS,PT2_MIN,PT2_MAX, ADC2_NBINS,0.0,ADC2_MAX)

// ------------------------------ CLI PARSING ------------------------------
OptionStore opt=new OptionStore("gain_MPV_calib_check")
opt.addCommand("process","Per–wire ADC gain calibration & check")

def cli=opt.getOptionParser("process")
cli.addOption("-nevent","")
cli.addOption("-beam","")
cli.addOption("-w2min","")
cli.addOption("-w2max","")
cli.addOption("-dphiHalf","")

cli.addOption("-vzmin","")
cli.addOption("-vzmax","")
cli.addOption("-fdonly","")

cli.addOption("-ptmin","")
cli.addOption("-ptmax","")

cli.addOption("-ctrlLo","")
cli.addOption("-ctrlHi","")

cli.addOption("-submode","")     // nnls / raw
cli.addOption("-skipFailed","")  // true / false

cli.addOption("-pt2min","")
cli.addOption("-pt2max","")
cli.addOption("-pt2bins","")
cli.addOption("-adc2max","")

cli.addOption("-fitLo","")
cli.addOption("-fitHi","")

cli.addOption("-csv","")         // CSV path
cli.addOption("-mode","")        // calib / check
cli.addOption("-species","")     // deut / prot
cli.addOption("-mode_valid","")  // "banana" => BANANA_ON
cli.addOption("-trackid","")     // force trackid

opt.parse(args)
if(opt.getCommand()!="process"){
  System.err.println("Usage: run-groovy gain_MPV_calib_check_final.groovy process [options] files.hipo ...")
  System.exit(1)
}

// input list
List<String> expandGlob(String pat){
  ArrayList<String> out=new ArrayList<>()
  if(pat==null) return out
  File f=new File(pat)
  if(f.exists() && f.isFile()){
    out.add(f.path)
    return out
  }
  if(pat.contains("*") || pat.contains("?")){
    File parent=f.getParentFile()
    if(parent==null) parent=new File(".")
    String rx="\\Q"+f.getName().replace("?", "\\E.\\Q").replace("*","\\E.*\\Q")+"\\E"
    def re=java.util.regex.Pattern.compile(rx)
    File[] list=parent.listFiles()
    if(list!=null){
      for(File ff : list){
        if(ff.isFile() && re.matcher(ff.getName()).matches()) out.add(ff.path)
      }
    }
  }
  return out
}

ArrayList<String> rawInputs=new ArrayList<>()
for(String s : cli.getInputList()) rawInputs.add(s)

ArrayList<String> files=new ArrayList<>()
for(String s : rawInputs){
  if(s==null) continue
  if(s.toLowerCase(Locale.ROOT).endsWith(".hipo")) files.add(s)
  else if(s.contains("*") || s.contains("?")) files.addAll(expandGlob(s))
}
LinkedHashSet<String> uniq=new LinkedHashSet<>(files)
files.clear()
files.addAll(uniq)
Iterator<String> itf=files.iterator()
while(itf.hasNext()){
  String f=itf.next()
  if(!(new File(f).isFile())) itf.remove()
}
if(files.isEmpty()){
  System.err.println("No .hipo input files.")
  System.exit(1)
}

// parse numeric/string options
String v
try{ v=cli.getOption("-nevent")?.getValue();  if(v!=null) MAXEV      = Long.parseLong(v.trim().replaceAll("[^0-9]","")) }catch(Exception e){}
try{ v=cli.getOption("-beam")?.getValue();    if(v!=null) EBEAM      = Double.parseDouble(v.trim()) }catch(Exception e){}
try{ v=cli.getOption("-w2min")?.getValue();   if(v!=null) W2_MIN     = Double.parseDouble(v.trim()) }catch(Exception e){}
try{ v=cli.getOption("-w2max")?.getValue();   if(v!=null) W2_MAX     = Double.parseDouble(v.trim()) }catch(Exception e){}
try{ v=cli.getOption("-dphiHalf")?.getValue();if(v!=null) DPHI_HALF  = Double.parseDouble(v.trim()) }catch(Exception e){}
try{ v=cli.getOption("-vzmin")?.getValue();   if(v!=null) VZ_MIN     = Double.parseDouble(v.trim()) }catch(Exception e){}
try{ v=cli.getOption("-vzmax")?.getValue();   if(v!=null) VZ_MAX     = Double.parseDouble(v.trim()) }catch(Exception e){}
try{ v=cli.getOption("-fdonly")?.getValue();  if(v!=null) FD_ONLY    = Boolean.parseBoolean(v.trim()) }catch(Exception e){}

try{ v=cli.getOption("-ptmin")?.getValue();   if(v!=null) PT_SLICE_MIN=Double.parseDouble(v.trim()) }catch(Exception e){}
try{ v=cli.getOption("-ptmax")?.getValue();   if(v!=null) PT_SLICE_MAX=Double.parseDouble(v.trim()) }catch(Exception e){}

try{ v=cli.getOption("-ctrlLo")?.getValue();  if(v!=null) CTRL_LO    = Double.parseDouble(v.trim()) }catch(Exception e){}
try{ v=cli.getOption("-ctrlHi")?.getValue();  if(v!=null) CTRL_HI    = Double.parseDouble(v.trim()) }catch(Exception e){}

try{ v=cli.getOption("-submode")?.getValue(); if(v!=null) SUBMODE    = v.trim().toLowerCase(Locale.ROOT) }catch(Exception e){}
try{ v=cli.getOption("-skipFailed")?.getValue(); if(v!=null) SKIP_FAILED_PRE=Boolean.parseBoolean(v.trim()) }catch(Exception e){}

try{ v=cli.getOption("-pt2min")?.getValue();  if(v!=null) PT2_MIN    = Double.parseDouble(v.trim()) }catch(Exception e){}
try{ v=cli.getOption("-pt2max")?.getValue();  if(v!=null) PT2_MAX    = Double.parseDouble(v.trim()) }catch(Exception e){}
try{ v=cli.getOption("-pt2bins")?.getValue(); if(v!=null) PT2_NBINS  = Math.max(1,Integer.parseInt(v.trim())) }catch(Exception e){}
try{
  v=cli.getOption("-adc2max")?.getValue()
  if(v!=null){
    ADC2_MAX=Double.parseDouble(v.trim())
    H2_PRE_ALL.getAxisY().setRange(ADC2_MIN,ADC2_MAX)
    H2_POST_ALL.getAxisY().setRange(0.0,ADC2_MAX)
    H2_PRE_L.getAxisY().setRange(ADC2_MIN,ADC2_MAX)
    H2_PRE_R.getAxisY().setRange(ADC2_MIN,ADC2_MAX)
    H2_POST_L.getAxisY().setRange(0.0,ADC2_MAX)
    H2_POST_R.getAxisY().setRange(0.0,ADC2_MAX)
  }
}catch(Exception e){}
try{ v=cli.getOption("-fitLo")?.getValue();   if(v!=null) FIT_LO     = Double.parseDouble(v.trim()) }catch(Exception e){}
try{ v=cli.getOption("-fitHi")?.getValue();   if(v!=null) FIT_HI     = Double.parseDouble(v.trim()) }catch(Exception e){}

try{ v=cli.getOption("-csv")?.getValue();     if(v!=null) CSV_OUT    = v.trim() }catch(Exception e){}
if(CSV_OUT==null || CSV_OUT.trim().isEmpty()){
  CSV_OUT = String.format(Locale.ROOT,"gains_MPV_%s.csv",SUBMODE)
}

try{
  v=cli.getOption("-mode")?.getValue()
  if(v!=null){
    String m=v.trim().toLowerCase(Locale.ROOT)
    if(m=="calib" || m=="check") MODE=m
  }
}catch(Exception e){}

try{
  v=cli.getOption("-species")?.getValue()
  if(v!=null){
    String sp=v.trim().toLowerCase(Locale.ROOT)
    if(sp=="deut" || sp=="prot") SPECIES=sp
  }
}catch(Exception e){}

try{
  v=cli.getOption("-mode_valid")?.getValue()
  if(v!=null){
    String mv=v.trim().toLowerCase(Locale.ROOT)
    BANANA_ON = mv.equals("banana")
  }
}catch(Exception e){}

// forced trackid
Integer FORCE_TRACK_ID=null
try{
  v=cli.getOption("-trackid")?.getValue()
  if(v!=null) FORCE_TRACK_ID=Integer.valueOf(v.trim())
}catch(Exception e){}

// ------------------------------ PASS–1: BUILD SIG/BKG ------------------------------
ProgressPrintout prog = new ProgressPrintout()
long seen=0L

for(String fn : files){
  HipoDataSource R=new HipoDataSource()
  try{ R.open(fn) }catch(Exception ex){
    System.err.println("Open fail "+fn+" : "+ex)
    continue
  }
  while(R.hasEvent()){
    DataEvent ev
    try{ ev=R.getNextEvent() }catch(Exception ex){ break }
    seen++

    RecP e = getElectronREC(ev)
    if(e==null){
      prog.updateStatus()
      if(MAXEV>0 && seen>=MAXEV) break
      else continue
    }

    double pt = Math.hypot((double)e.px,(double)e.py)
    if(pt<PT_SLICE_MIN || pt>PT_SLICE_MAX){
      prog.updateStatus()
      if(MAXEV>0 && seen>=MAXEV) break
      else continue
    }

    double phi_e=phiDeg(e.px,e.py)
    KFRec kBack = bestKF_BackToBack(ev,phi_e)
    KFRec kHeavy= bestKF_Heavy(ev)
    if(kBack.idx<0 || kHeavy.idx<0){
      prog.updateStatus()
      if(MAXEV>0 && seen>=MAXEV) break
      else continue
    }

    int assocId=(FORCE_TRACK_ID!=null)? FORCE_TRACK_ID.intValue(): kHeavy.trackid
    if(assocId<0){
      prog.updateStatus()
      if(MAXEV>0 && seen>=MAXEV) break
      else continue
    }
    AssocSets AS = buildAssocSetsForTrackId(ev,assocId)

    double W  = W_from_e(EBEAM,e)
    double W2 = (Double.isNaN(W)? Double.NaN : W*W)
    boolean okW =(!Double.isNaN(W2) && between(W2,W2_MIN,W2_MAX))
    boolean okDP=(Math.abs(dphi0to360(phi_e,kBack.phiDeg)-180.0d) <= DPHI_HALF)

    long sumTRK=0L
    if(ev.hasBank("AHDC::adc")){
      DataBank a0=ev.getBank("AHDC::adc")
      for(int i=0;i<a0.rows();i++){
        int A=0
        try{ A=a0.getInt("ADC",i) }catch(Exception ign){}
        if(A<=0) continue
        int Lraw=a0.getInt("layer",i)
        int c   =a0.getInt("component",i)
        int sl=Lraw/10
        int l =Lraw%10
        if(AS.slw.contains(new SLW(sl,l,c))) sumTRK+=(long)A
      }
    }
    try{
      DataBank k=ev.getBank("AHDC::kftrack")
      int sadc=k.getInt("sum_adc",kHeavy.idx)
      if(sadc>0) sumTRK=(long)sadc
    }catch(Exception ign){}

    boolean baseElastic = okW && okDP

    boolean speciesGate = true
    if(BANANA_ON){
      if(SPECIES.equalsIgnoreCase("deut"))      speciesGate = inBanana(pt,sumTRK)
      else if(SPECIES.equalsIgnoreCase("prot")) speciesGate = inProton(pt,sumTRK)
    }
    boolean ELASTIC = baseElastic && speciesGate

    if(!ev.hasBank("AHDC::adc")){
      prog.updateStatus()
      if(MAXEV>0 && seen>=MAXEV) break
      else continue
    }
    DataBank a=ev.getBank("AHDC::adc")
    Set<String> wfGoodSet=wfExplicitGood(ev)

    for(int i=0;i<a.rows();i++){
      int A=0
      try{ A=a.getInt("ADC",i) }catch(Exception ign){}
      if(A<=0) continue
      if(!wfPassForAdcRow(a,i,wfGoodSet)) continue

      int Lraw=a.getInt("layer",i)
      int c   =a.getInt("component",i)
      int sl=Lraw/10
      int l =Lraw%10
      int s =a.getInt("sector",i)
      if(!AS.slw.contains(new SLW(sl,l,c))) continue

      double xadc=clampADC((double)A,ADC_MAX)
      WireKey wk=new WireKey(s,Lraw,c)
      PairH   ph=getPair(wk)
      if(ELASTIC){
        ph.sig.fill(xadc)
        SUM_SIG.fill(xadc)
      }else{
        ph.bkg.fill(xadc)
        SUM_BKG.fill(xadc)
      }
    }
    prog.updateStatus()
    if(MAXEV>0 && seen>=MAXEV) break
  }
  try{ R.close() }catch(Exception ign){}
}

// ------------------------------ NNLS α SEARCH ------------------------------
class NNLSAlpha{
  static double objective(H1F sig,H1F bkg,double alpha,double ctrlLo,double ctrlHi){
    def ax=sig.getAxis()
    int n=ax.getNBins()
    double obj=0.0
    for(int b=0;b<n;b++){
      double xC=ax.getBinCenter(b)
      if(xC<ctrlLo || xC>ctrlHi) continue
      double S=sig.getBinContent(b)
      double B=bkg.getBinContent(b)
      if(B<=0 && S<=0) continue
      double d=alpha*B - S
      if(d<=0) continue // only penalize where SUB would be negative
      double var = S + alpha*alpha*B + 1.0
      obj += (d*d)/var
    }
    return obj
  }

  static double searchAlpha(H1F sig,H1F bkg,double aGuess,
                            double aMin,double aMax,
                            double ctrlLo,double ctrlHi){
    double lo=Math.max(aMin, aGuess/5.0)
    double hi=Math.min(aMax, aGuess*5.0)
    if(!(lo<hi)){ lo=aMin; hi=aMax }
    double phi=(Math.sqrt(5.0)-1.0)/2.0
    double c=hi - phi*(hi-lo)
    double d=lo + phi*(hi-lo)
    double fc=objective(sig,bkg,c,ctrlLo,ctrlHi)
    double fd=objective(sig,bkg,d,ctrlLo,ctrlHi)
    for(int it=0; it<64; it++){
      if(fc>fd){
        lo=c
        c=d
        fc=fd
        d=lo + phi*(hi-lo)
        fd=objective(sig,bkg,d,ctrlLo,ctrlHi)
      }else{
        hi=d
        d=c
        fd=fc
        c=hi - phi*(hi-lo)
        fc=objective(sig,bkg,c,ctrlLo,ctrlHi)
      }
      if(Math.abs(hi-lo)<1e-4*Math.max(1.0,0.5*(Math.abs(lo)+Math.abs(hi)))) break
    }
    double a=(fc<fd? c:d)
    return Math.max(aMin,Math.min(aMax,a))
  }
}

// ------------------------------ α, SUB, α·BKG, SUM_SUB ------------------------------
AlphaCtrlRes arSUM=alphaFromControl(SUM_SIG,SUM_BKG,CTRL_LO,CTRL_HI)
double A_glob = (arSUM.alphaPref>0.0 ? arSUM.alphaPref
               : (arSUM.alphaLSQ>0.0 ? arSUM.alphaLSQ
               : (arSUM.alphaRatio>0.0 ? arSUM.alphaRatio : 1.0)))
A_glob = clampAlpha(A_glob)

// per–wire
for(Map.Entry<WireKey,PairH> e2 : histMap.entrySet()){
  WireKey wk=e2.getKey()
  PairH   ph=e2.getValue()

  AlphaCtrlRes ar=alphaFromControl(ph.sig,ph.bkg,CTRL_LO,CTRL_HI)
  double a0=(ar.alphaPref>0.0 ? ar.alphaPref
             : (ar.alphaLSQ>0.0 ? ar.alphaLSQ
             : (ar.alphaRatio>0.0 ? ar.alphaRatio : A_glob)))
  a0=clampAlpha(a0)
  double aWire=(SUBMODE.equals("nnls")
                 ? NNLSAlpha.searchAlpha(ph.sig,ph.bkg,a0,ALPHA_MIN,ALPHA_MAX,CTRL_LO,CTRL_HI)
                 : a0)
  aWire=clampAlpha(aWire)
  alphaMap.put(wk,aWire)

  int nBins=ph.sub.getAxis().getNBins()
  for(int b=0;b<nBins;b++){
    double S=ph.sig.getBinContent(b)
    double B=ph.bkg.getBinContent(b)
    double y_raw=S - aWire*B
    double y = (SUBMODE.equals("nnls") ? Math.max(0.0,y_raw) : y_raw)
    ph.sub.setBinContent(b,y)
    double err=Math.sqrt(Math.max(1.0,S + aWire*aWire*B))
    try{ ph.sub.setBinError(b,err) }catch(Exception ign){}
  }

  // α·BKG for overlay
  int nbB=ph.bkg.getAxis().getNBins()
  double xloB=ph.bkg.getAxis().min()
  double xhiB=ph.bkg.getAxis().max()
  if(nbB<=0){ nbB=ADC_NBINS; xloB=0.0; xhiB=ADC_MAX }
  H1F hScaled=new H1F("bkg_scaled_"+wk.toString().replace(' ','_'),
                      "α·BKG — "+wk.toString()+";ADC;Counts",
                      nbB,xloB,xhiB)
  for(int b=0;b<nbB;b++){
    double B=ph.bkg.getBinContent(b)
    if(B<=0.0){
      hScaled.setBinContent(b,0.0)
      try{ hScaled.setBinError(b,0.0) }catch(Exception ign){}
      continue
    }
    double y = aWire*B
    double err=Math.sqrt(Math.max(1.0,aWire*aWire*B))
    hScaled.setBinContent(b,y)
    try{ hScaled.setBinError(b,err) }catch(Exception ign){}
  }
  hScaled.setLineColor(2)
  scaledBkgMap.put(wk,hScaled)
}

// SUM_SUB
int nbSS=SUM_SUB.getAxis().getNBins()
for(int b=0;b<nbSS;b++){
  double S=SUM_SIG.getBinContent(b)
  double B=SUM_BKG.getBinContent(b)
  double y_raw=S - A_glob*B
  double y = (SUBMODE.equals("nnls") ? Math.max(0.0,y_raw) : y_raw)
  SUM_SUB.setBinContent(b,y)
  double err=Math.sqrt(Math.max(1.0,S + A_glob*A_glob*B))
  try{ SUM_SUB.setBinError(b,err) }catch(Exception ign){}
}

// ------------------------------ GLOBAL REF MPV ------------------------------
FitResultGlobal frSum = fitGlobalLandau(SUM_SUB,FIT_LO,FIT_HI)
double MPV_REF    = Double.NaN
double MPV_REF_ERR= 0.0
if(frSum.ok){
  MPV_REF    = frSum.mpv
  MPV_REF_ERR= frSum.mpvErr
  System.out.printf(Locale.ROOT,
    "Global SUM_SUB Landau: MPV_REF = %.1f ± %.1f ADC (chi2/ndf = %.1f/%d)%n",
    frSum.mpv,frSum.mpvErr,frSum.chi2,frSum.ndf)
}else{
  System.out.println("WARNING: Global SUM_SUB Landau not fitted (too few entries).")
}

// ------------------------------ CSV GAINS (for CHECK mode) ------------------------------
class CSVGainEntry{
  double alpha
  double mpvPre, mpvPreErr, chi2ndfPre
  double mpvPost,mpvPostErr,chi2ndfPost
  double gain,gainErr
  boolean valid=false
}
Map<WireKey,CSVGainEntry> loadGainsCSV(String path){
  Map<WireKey,CSVGainEntry> map=new LinkedHashMap<>()
  if(path==null) return map
  try{
    File f=new File(path)
    if(!f.isFile()){
      System.err.println("CHECK mode: CSV '"+path+"' not found, using gain=1.")
      return map
    }
    f.withReader("UTF-8"){ reader ->
      String line
      boolean first=true
      while((line=reader.readLine())!=null){
        line=line.trim()
        if(line.isEmpty()) continue
        if(first){
          first=false
          if(line.toLowerCase(Locale.ROOT).startsWith("sector")) continue
        }
        String[] t=line.split(",")
        if(t.length<16) continue
        int s   = Integer.parseInt(t[0].trim())
        int Lraw= Integer.parseInt(t[3].trim())
        int c   = Integer.parseInt(t[4].trim())
        WireKey wk=new WireKey(s,Lraw,c)

        CSVGainEntry ge=new CSVGainEntry()
        ge.alpha      = parseDoubleSafe(t[6])
        ge.mpvPre     = parseDoubleSafe(t[8])
        ge.mpvPreErr  = parseDoubleSafe(t[9])
        ge.chi2ndfPre = parseDoubleSafe(t[10])
        ge.mpvPost    = parseDoubleSafe(t[11])
        ge.mpvPostErr = parseDoubleSafe(t[12])
        ge.chi2ndfPost= parseDoubleSafe(t[13])
        ge.gain       = parseDoubleSafe(t[14])
        ge.gainErr    = parseDoubleSafe(t[15])
        ge.valid=(ge.gain>0.0 && !Double.isNaN(ge.gain))
        map.put(wk,ge)
      }
    }
    System.out.println("CHECK mode: loaded "+map.size()+" gain rows from "+path)
  }catch(Exception ex){
    System.err.println("CHECK mode: error reading CSV '"+path+"' : "+ex)
  }
  return map
}
Map<WireKey,CSVGainEntry> csvGainMap = (MODE.equals("check")
                                        ? loadGainsCSV(CSV_OUT)
                                        : new LinkedHashMap<WireKey,CSVGainEntry>())

// ------------------------------ PER–WIRE FITS / GAINS ------------------------------
class FitPack{
  FitResult1D pre
  FitResult1D post
  double g=Double.NaN
  double ge=Double.NaN
}
@Field Map<WireKey,FitPack> fitPackMap = new LinkedHashMap<>()

for(Map.Entry<WireKey,PairH> e3 : histMap.entrySet()){
  WireKey wk=e3.getKey()
  PairH   ph=e3.getValue()
  H1F hSub=ph.sub
  FitPack pack=new FitPack()
  H1F hCorr=null

  if(MODE.equals("calib")){
    // pre–gain fit on SUB
    pack.pre = fitLandau1D(hSub,FIT_LO,FIT_HI,"pre_"+wk.toString().replace(' ','_'))
    if(pack.pre!=null && pack.pre.ok && MPV_REF>0.0){
      pack.g = MPV_REF / pack.pre.mpv
      double rel_ref  = (MPV_REF_ERR>0.0 ? MPV_REF_ERR/MPV_REF : 0.0)
      double rel_wire = (pack.pre.mpv>0.0 && pack.pre.mpvErr>0.0
                         ? pack.pre.mpvErr/pack.pre.mpv
                         : 0.0)
      pack.ge = pack.g*Math.sqrt(rel_ref*rel_ref + rel_wire*rel_wire)
    }
    hCorr=buildGainCorrHist(hSub,wk,pack.g)
    pack.post=fitLandau1D(hCorr,FIT_LO,FIT_HI,"post_"+wk.toString().replace(' ','_'))
  }else{ // CHECK
    CSVGainEntry geCSV=csvGainMap.get(wk)
    double gCSV=(geCSV!=null && geCSV.valid)? geCSV.gain : Double.NaN
    pack.g  = gCSV
    pack.ge = (geCSV!=null ? geCSV.gainErr : Double.NaN)

    pack.pre  = new FitResult1D()
    pack.post = new FitResult1D()
    if(geCSV!=null){
      pack.pre.mpv     = geCSV.mpvPre
      pack.pre.mpvErr  = Math.max(0.0,geCSV.mpvPreErr)
      pack.pre.ok      = (!Double.isNaN(pack.pre.mpv) && pack.pre.mpv>0.0)

      pack.post.mpv    = geCSV.mpvPost
      pack.post.mpvErr = Math.max(0.0,geCSV.mpvPostErr)
      pack.post.ok     = (!Double.isNaN(pack.post.mpv) && pack.post.mpv>0.0)
    }
    hCorr=buildGainCorrHist(hSub,wk,gCSV)
  }

  subGainMap.put(wk,hCorr)
  fitPackMap.put(wk,pack)
}

// ------------------------------ GUI #1 : SUB, SUBGAIN, SUM, GRAPHS ------------------------------
ArrayList<WireKey> allKeys=new ArrayList<>(histMap.keySet())
Collections.sort(allKeys,new Comparator<WireKey>(){
  int compare(WireKey a,WireKey b){
    if(a.s!=b.s)       return a.s-b.s
    if(a.Lraw!=b.Lraw) return a.Lraw-b.Lraw
    return a.c-b.c
  }
})
int total=allKeys.size()
int pages=(int)Math.ceil(total/(double)PAGE_SIZE())

ArrayList<String> tabs1=new ArrayList<>()
for(int p=1;p<=pages;p++){
  tabs1.add(String.format("SUB [%s] [p%d/%d]",SUBMODE,p,pages))
  tabs1.add(String.format("SUB gain [p%d/%d]",p,pages))
}
tabs1.add("SUM & REF")
tabs1.add("Wire MPV & Gain")

EmbeddedCanvasTabbed canv1=new EmbeddedCanvasTabbed(tabs1.toArray(new String[0]))

def drawSubPage={ String name, boolean post, int pageIdx ->
  def cx=canv1.getCanvas(name)
  cx.divide(DRAW_COLS,DRAW_ROWS)
  int start=(pageIdx-1)*PAGE_SIZE()
  int end  =Math.min(start+PAGE_SIZE(),total)
  int pad=0
  for(int i=0;i<end-start;i++){
    int idx=start+i
    if(idx>=total) break
    WireKey wk=allKeys.get(idx)
    FitPack pack=fitPackMap.get(wk)
    if(SKIP_FAILED_PRE && !post && (pack==null || pack.pre==null || !pack.pre.ok)){
      pad++
      continue
    }
    H1F h = post ? subGainMap.get(wk) : histMap.get(wk).sub
    if(h==null){ pad++; continue }
    cx.cd(pad)
    h.setLineColor(post?6:4)
    cx.draw(h)
    if(MODE.equals("calib") && pack!=null){
      FitResult1D fr = post ? pack.post : pack.pre
      if(fr!=null && fr.ok && fr.func!=null) cx.draw(fr.func,"same")
    }
    YSym yr=symmetricY(h,1.20)
    double xmin=h.getAxis().min()
    double xmax=h.getAxis().max()
    setPadRangesSafe(cx,pad,xmin,xmax,yr.ymin,yr.ymax)
    pad++
  }
}

for(int p=1;p<=pages;p++){
  drawSubPage(String.format("SUB [%s] [p%d/%d]",SUBMODE,p,pages),false,p)
  drawSubPage(String.format("SUB gain [p%d/%d]",p,pages),true,p)
}

// SUM & REF
def csum=canv1.getCanvas("SUM & REF")
csum.divide(1,1)
csum.cd(0)
csum.draw(SUM_SIG)
csum.draw(SUM_BKG)
csum.draw(SUM_SUB)
if(frSum!=null && frSum.ok && frSum.func!=null) csum.draw(frSum.func,"same")

// MPV / gain graphs
GraphErrors gPre = new GraphErrors("MPV_pre_vs_wire")
GraphErrors gPost= new GraphErrors("MPV_post_vs_wire")
GraphErrors gGain= new GraphErrors("Gain_vs_wire")
int idxWire=0
for(WireKey wk : allKeys){
  FitPack pack=fitPackMap.get(wk)
  double x=(double)idxWire
  if(pack!=null){
    if(pack.pre!=null && pack.pre.ok){
      double ep=(pack.pre.mpvErr>0.0? pack.pre.mpvErr:0.0)
      gPre.addPoint(x,pack.pre.mpv,0.0,ep)
      if(!Double.isNaN(pack.g) && pack.g>0.0){
        double eg=(pack.ge>0.0? pack.ge:0.0)
        gGain.addPoint(x,pack.g,0.0,eg)
      }
    }
    if(pack.post!=null && pack.post.ok){
      double epo=(pack.post.mpvErr>0.0? pack.post.mpvErr:0.0)
      gPost.addPoint(x,pack.post.mpv,0.0,epo)
    }
  }
  idxWire++
}
[gPre,gPost,gGain].each{ gr ->
  gr.setMarkerStyle(2)
  gr.setMarkerSize(4)
}

// Graph styling
gPre.setTitle("MPV_pre vs wire;wire index;MPV_pre (ADC)")
gPost.setTitle("MPV_post vs wire;wire index;MPV_post (ADC)")
gGain.setTitle("Gain vs wire;wire index;gain")

gPre.setMarkerColor(1); gPre.setLineColor(1)
gPost.setMarkerColor(2); gPost.setLineColor(2)
gGain.setMarkerColor(3); gGain.setLineColor(3)

def cwg=canv1.getCanvas("Wire MPV & Gain")
cwg.divide(2,2)
cwg.cd(0); cwg.draw(gPre)
cwg.cd(1); cwg.draw(gPost)
cwg.cd(2); cwg.draw(gGain)
try{ cwg.getPad(2).getAxisFrame().getAxisY().setRange(0.7,1.3) }catch(Exception ign){}

// ------------------------------ CSV OUTPUT (CALIB) ------------------------------
if(MODE.equals("calib")){
  try{
    File csvFile=new File(CSV_OUT)
    csvFile.withPrintWriter("UTF-8"){ pw ->
      pw.println("sector,superlayer,layer,Lraw,component,wireIndex,alpha,submode,"+
                 "mpv_pre,mpv_pre_err,chi2ndf_pre,"+
                 "mpv_post,mpv_post_err,chi2ndf_post,"+
                 "gain,gain_err")
      int widx=0
      for(WireKey wk : allKeys){
        FitPack pack=fitPackMap.get(wk)
        if(pack==null || pack.pre==null || !pack.pre.ok ||
           Double.isNaN(pack.g) || pack.g<=0.0){
          widx++
          continue
        }
        double alpha=(alphaMap.containsKey(wk) && alphaMap.get(wk)!=null
                      ? alphaMap.get(wk)
                      : Double.NaN)
        int SL = wk.Lraw/10
        int L  = wk.Lraw%10
        double chi2ndfPre = (!Double.isNaN(pack.pre.chi2) && pack.pre.ndf>0
                              ? pack.pre.chi2/pack.pre.ndf
                              : Double.NaN)
        double chi2ndfPost=(pack.post!=null && pack.post.ok &&
                            !Double.isNaN(pack.post.chi2) && pack.post.ndf>0
                              ? pack.post.chi2/pack.post.ndf
                              : Double.NaN)
        double mpvPost    =(pack.post!=null && pack.post.ok? pack.post.mpv:Double.NaN)
        double mpvPostErr =(pack.post!=null && pack.post.ok? pack.post.mpvErr:Double.NaN)

        pw.printf(Locale.ROOT,
          "%d,%d,%d,%d,%d,%d,%.6f,%s,"+
          "%.3f,%.3f,%.3f,"+
          "%.3f,%.3f,%.3f,"+
          "%.5f,%.5f%n",
          wk.s,SL,L,wk.Lraw,wk.c,widx,
          alpha,SUBMODE,
          pack.pre.mpv,pack.pre.mpvErr,chi2ndfPre,
          mpvPost,mpvPostErr,chi2ndfPost,
          pack.g,pack.ge)
        widx++
      }
    }
    println "CALIB mode: gain CSV written to "+CSV_OUT
  }catch(Exception ex){
    System.err.println("ERROR writing CSV '"+CSV_OUT+"': "+ex)
  }
}else{
  println "CHECK mode: using gains from CSV = "+CSV_OUT
}

// show GUI #1
try{
  JFrame f1=new JFrame("AHDC per–wire gain calibration — MODE="+MODE+" SUB="+SUBMODE+" SPECIES="+SPECIES)
  f1.setSize(1600,900)
  f1.add(canv1)
  f1.setVisible(true)
  f1.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
}catch(Exception ign){}

// ------------------------------ PASS–2: pT–ADC 2D WITH GAINS ------------------------------
ProgressPrintout prog2=new ProgressPrintout()
long seen2=0L

for(String fn : files){
  HipoDataSource R=new HipoDataSource()
  try{ R.open(fn) }catch(Exception ex){
    System.err.println("Open fail "+fn+" (pass2): "+ex)
    continue
  }
  while(R.hasEvent()){
    DataEvent ev
    try{ ev=R.getNextEvent() }catch(Exception ex){ break }
    seen2++

    RecP e=getElectronREC(ev)
    if(e==null){ prog2.updateStatus(); continue }

    double pt = Math.hypot((double)e.px,(double)e.py)
    if(pt<PT_SLICE_MIN || pt>PT_SLICE_MAX){ prog2.updateStatus(); continue }

    double phi_e=phiDeg(e.px,e.py)
    KFRec kBack = bestKF_BackToBack(ev,phi_e)
    KFRec kHeavy= bestKF_Heavy(ev)
    if(kBack.idx<0 || kHeavy.idx<0){ prog2.updateStatus(); continue }

    int assocId=(FORCE_TRACK_ID!=null)? FORCE_TRACK_ID.intValue(): kHeavy.trackid
    if(assocId<0){ prog2.updateStatus(); continue }
    AssocSets AS=buildAssocSetsForTrackId(ev,assocId)

    double W=W_from_e(EBEAM,e)
    double W2=(Double.isNaN(W)? Double.NaN : W*W)
    boolean okW =(!Double.isNaN(W2) && between(W2,W2_MIN,W2_MAX))
    boolean okDP=(Math.abs(dphi0to360(phi_e,kBack.phiDeg)-180.0d)<=DPHI_HALF)

    long sumTRK=0L
    if(ev.hasBank("AHDC::adc")){
      DataBank a0=ev.getBank("AHDC::adc")
      for(int i=0;i<a0.rows();i++){
        int A=0
        try{ A=a0.getInt("ADC",i) }catch(Exception ign){}
        if(A<=0) continue
        int Lraw=a0.getInt("layer",i)
        int c   =a0.getInt("component",i)
        int sl=Lraw/10
        int l =Lraw%10
        if(AS.slw.contains(new SLW(sl,l,c))) sumTRK+=(long)A
      }
    }
    try{
      DataBank k=ev.getBank("AHDC::kftrack")
      int sadc=k.getInt("sum_adc",kHeavy.idx)
      if(sadc>0) sumTRK=(long)sadc
    }catch(Exception ign){}

    boolean baseElastic = okW && okDP
    boolean speciesGate = true
    if(BANANA_ON){
      if(SPECIES.equalsIgnoreCase("deut"))      speciesGate = inBanana(pt,sumTRK)
      else if(SPECIES.equalsIgnoreCase("prot")) speciesGate = inProton(pt,sumTRK)
    }
    boolean ELASTIC=baseElastic && speciesGate
    if(!ELASTIC){ prog2.updateStatus(); continue }

    if(!ev.hasBank("AHDC::adc")){ prog2.updateStatus(); continue }
    DataBank a=ev.getBank("AHDC::adc")
    Set<String> wfGoodSet=wfExplicitGood(ev)

    for(int i=0;i<a.rows();i++){
      int A=0
      try{ A=a.getInt("ADC",i) }catch(Exception ign){}
      if(A<=0) continue
      if(!wfPassForAdcRow(a,i,wfGoodSet)) continue

      int Lraw=a.getInt("layer",i)
      int c   =a.getInt("component",i)
      int sl=Lraw/10
      int l =Lraw%10
      int s =a.getInt("sector",i)
      if(!AS.slw.contains(new SLW(sl,l,c))) continue

      double adcRaw=clampADC((double)A,ADC_MAX)
      double gWire=1.0
      WireKey wk=new WireKey(s,Lraw,c)
      FitPack pk=fitPackMap.get(wk)
      if(pk!=null && !Double.isNaN(pk.g) && pk.g>0.0) gWire=pk.g
      double adcCorr=clampADC(adcRaw*gWire,ADC_MAX)

      H2_PRE_ALL.fill(pt,adcRaw)
      H2_POST_ALL.fill(pt,adcCorr)

      int ord=-1
      boolean hasOrd=false
      try{
        ord=a.getByte("order",i)
        hasOrd=true
        HAS_ORDER=true
      }catch(Exception ignOrd){}
      if(hasOrd){
        if(ord==0){
          H2_PRE_L.fill(pt,adcRaw)
          H2_POST_L.fill(pt,adcCorr)
        }else if(ord==1){
          H2_PRE_R.fill(pt,adcRaw)
          H2_POST_R.fill(pt,adcCorr)
        }
      }
    }
    prog2.updateStatus()
  }
  try{ R.close() }catch(Exception ign){}
}

// ------------------------------ GUI #2: OVERLAYS & pT–ADC ------------------------------
ArrayList<String> tabs2=new ArrayList<>()
for(int p=1;p<=pages;p++){
  tabs2.add(String.format("Overlay pre/post [p%d/%d]",p,pages))
  tabs2.add(String.format("SIG vs αBKG [p%d/%d]",p,pages))
}
tabs2.add("pT–ADC (pre/post)")
if(HAS_ORDER) tabs2.add("pT–ADC (L/R × pre/post)")

EmbeddedCanvasTabbed canv2=new EmbeddedCanvasTabbed(tabs2.toArray(new String[0]))

def drawOverlayPage={ String name,int pageIdx ->
  def cx=canv2.getCanvas(name)
  cx.divide(DRAW_COLS,DRAW_ROWS)
  int start=(pageIdx-1)*PAGE_SIZE()
  int end  =Math.min(start+PAGE_SIZE(),total)
  int pad=0
  for(int i=0;i<end-start;i++){
    int idx=start+i
    if(idx>=total) break
    WireKey wk=allKeys.get(idx)
    FitPack pack=fitPackMap.get(wk)
    if(SKIP_FAILED_PRE && (pack==null || pack.pre==null || !pack.pre.ok)){
      pad++
      continue
    }
    H1F hPre = histMap.get(wk).sub
    H1F hPost= subGainMap.get(wk)
    cx.cd(pad)
    if(hPre!=null){
      hPre.setLineColor(4)
      cx.draw(hPre)
    }
    if(hPost!=null){
      hPost.setLineColor(6)
      cx.draw(hPost,"same")
    }
    if(MODE.equals("calib") && pack!=null){
      if(pack.pre!=null && pack.pre.ok && pack.pre.func!=null)  cx.draw(pack.pre.func,"same")
      if(pack.post!=null && pack.post.ok && pack.post.func!=null)cx.draw(pack.post.func,"same")
    }
    if(hPre!=null){
      YSym yr=symmetricY(hPre,1.25)
      double xmin=hPre.getAxis().min()
      double xmax=hPre.getAxis().max()
      setPadRangesSafe(cx,pad,xmin,xmax,yr.ymin,yr.ymax)
    }
    pad++
  }
}

def drawSigAlphaPage={ String name,int pageIdx ->
  def cx=canv2.getCanvas(name)
  cx.divide(DRAW_COLS,DRAW_ROWS)
  int start=(pageIdx-1)*PAGE_SIZE()
  int end  =Math.min(start+PAGE_SIZE(),total)
  int pad=0
  for(int i=0;i<end-start;i++){
    int idx=start+i
    if(idx>=total) break
    WireKey wk=allKeys.get(idx)
    PairH ph=histMap.get(wk)
    H1F hSig = (ph!=null ? ph.sig : null)
    H1F hSca = scaledBkgMap.get(wk)
    if(hSig==null || hSca==null){
      pad++
      continue
    }
    cx.cd(pad)
    hSig.setLineColor(1)
    cx.draw(hSig)
    hSca.setLineColor(2)
    cx.draw(hSca,"same")

    double xmin=hSig.getAxis().min()
    double xmax=hSig.getAxis().max()
    double ymax=0.0
    int nb=hSig.getAxis().getNBins()
    for(int b=0;b<nb;b++){
      ymax=Math.max(ymax,hSig.getBinContent(b))
      ymax=Math.max(ymax,hSca.getBinContent(b))
    }
    ymax=(ymax<=0.0 ? 1.0 : ymax*1.2)
    setPadRangesSafe(cx,pad,xmin,xmax,0.0,ymax)
    pad++
  }
}

for(int p=1;p<=pages;p++){
  drawOverlayPage(String.format("Overlay pre/post [p%d/%d]",p,pages),p)
  drawSigAlphaPage(String.format("SIG vs αBKG [p%d/%d]",p,pages),p)
}

if(tabs2.contains("pT–ADC (pre/post)")){
  def c2=canv2.getCanvas("pT–ADC (pre/post)")
  c2.divide(2,1)
  c2.cd(0); c2.draw(H2_PRE_ALL)
  c2.cd(1); c2.draw(H2_POST_ALL)
}

if(HAS_ORDER && tabs2.contains("pT–ADC (L/R × pre/post)")){
  def cLR=canv2.getCanvas("pT–ADC (L/R × pre/post)")
  cLR.divide(2,2)
  cLR.cd(0); cLR.draw(H2_PRE_L)
  cLR.cd(1); cLR.draw(H2_PRE_R)
  cLR.cd(2); cLR.draw(H2_POST_L)
  cLR.cd(3); cLR.draw(H2_POST_R)
}

try{
  JFrame f2=new JFrame("AHDC gain QA — overlays & pT–ADC — MODE="+MODE+" SPECIES="+SPECIES)
  f2.setSize(1600,900)
  f2.add(canv2)
  f2.setVisible(true)
  f2.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
}catch(Exception ign){}

System.out.printf(Locale.ROOT,
  "Done. MODE=%s SUBMODE=%s SPECIES=%s wires=%d; MPV_REF=%.1f ± %.1f ADC; CSV=%s%n",
  MODE,SUBMODE,SPECIES,total,MPV_REF,MPV_REF_ERR,CSV_OUT)
