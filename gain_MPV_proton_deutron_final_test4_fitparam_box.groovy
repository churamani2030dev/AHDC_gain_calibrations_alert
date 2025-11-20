// gain_MPV_proton_deutron_final_test4_fitparam_box.groovy
// Per-wire ADC calibration with robust background subtraction,
// Landau MPV fits (pre/post), global REF from SUM_SUB, gain + errors,
// 10-plots-per-canvas across all wires (SUB pre, SUB gain), MPV/gain trends,
// Separate GUI: per-wire overlay SUB (pre, blue) vs SUB gain-corr (magenta)
// Separate GUI: global pT–ADC 2D (SUB-only) before/after in one 1×2 canvas;
// and, if 'order' exists, a 2×2 tab (Left/Right × pre/post).
//
// **Update in this version**: draw the actual fitted F1D with setOptStat(1111)
// so a stats box appears on each pad that has a fit. Nothing else changed.
//
// Run example:
//   run-groovy gain_MPV_proton_deutron_final_test4_fitparam_box.groovy process \
//     -trackid 1 -nevent 5000000000 -mode_valid banana \
//     -ptmin 0.20 -ptmax 0.45 \
//     -ctrlLo 80 -ctrlHi 480 \
//     /path/to/*.hipo

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

// ------------------------------ Config ------------------------------
@Field double EBEAM=2.24d, M_D=1.875612d
@Field double W2_MIN=3.46d, W2_MAX=3.67d, DPHI_HALF=10.0d
@Field boolean BANANA_ON=false
@Field double BAN_SCALE=1000.0d, BAN_C0=6.0d, BAN_C1=-35.0d, BAN_HALF=1.5d
@Field int    KF_NHITS_MIN=8
@Field double KF_CHI2_MAX=30.0d
@Field double VZ_MIN=-20.0d, VZ_MAX=+10.0d
@Field boolean FD_ONLY=true
@Field double PT_SLICE_MIN=0.20d, PT_SLICE_MAX=0.45d      // used for per-wire MPV extraction
@Field long   MAXEV=-1L

@Field int    ADC_NBINS=220
@Field double ADC_MAX=4500.0d

@Field double CTRL_LO=80.0d, CTRL_HI=480.0d, ALPHA_MIN=0.001d, ALPHA_MAX=5.0d
@Field int    CTRL_MIN_BINS=4
@Field double CTRL_MIN_BKG_SUM=20.0d

@Field boolean DO_PER_WIRE_FITS=true
@Field boolean DRAW_GREEN_FITS=true
@Field boolean ZERO_FLOOR_SUB=true

// Landau fit window for per-wire & global SUB spectra
@Field double FIT_LO=450.0d,   FIT_HI=2000.0d

// 2D pT–ADC (SUB-only) before/after: binning + ranges (unchanged here)
@Field int    PT2_NBINS=80
@Field double PT2_MIN=0.15d,   PT2_MAX=1.20d
@Field int    ADC2_NBINS=220
@Field double ADC2_MIN=0.0d,   ADC2_MAX=4500.0d

// Layout: 10 pads per canvas (5x2)
@Field int DRAW_COLS=5, DRAW_ROWS=2
int PAGE_SIZE(){ return DRAW_COLS*DRAW_ROWS }

// -------------------------- Helpers ---------------------------------
static double deg0to360(double a){ double x=a%360.0d; return (x<0.0d)? x+360.0d : x }
static double phiDeg(float px,float py){ return deg0to360(Math.toDegrees(Math.atan2((double)py,(double)px))) }
static double dphi0to360(double pe,double pk){ double x=((pe - pk)%360.0d); return (x<0.0d)? x+360.0d : x }
static boolean between(double x,double lo,double hi){ return (x>=lo && x<=hi) }
static int    Lenc(int sl,int l){ return 10*sl + l }
static double clampADC(double A, double max){ return Math.min(max-1e-6, A) }
boolean inBanana(double pt, long sumADC_trk){ double y=((double)sumADC_trk)/BAN_SCALE; double c=BAN_C0 + BAN_C1*(pt-0.26d); return Math.abs(y-c)<=BAN_HALF }
double clampAlpha(double a){ if(Double.isNaN(a)||Double.isInfinite(a)) return ALPHA_MIN; return Math.max(ALPHA_MIN,Math.min(ALPHA_MAX,a)) }

class YSym{ double ymin,ymax; YSym(double a,double b){ymin=a; ymax=b} }
YSym symmetricY(H1F h,double pad){
  int n=h.getAxis().getNBins(); double ymin=0,ymax=0
  for(int b=0;b<n;b++){ double y=h.getBinContent(b); if(b==0){ymin=y;ymax=y}else{ if(y<ymin) ymin=y; if(y>ymax) ymax=y } }
  double a=Math.max(Math.abs(ymin),Math.abs(ymax)); a=(a<=0?1.0:a*pad); return new YSym(-a,+a)
}
void setPadRangesSafe(def canvas,int padIndex,double xmin,double xmax,double ymin,double ymax){
  try{ canvas.getPad(padIndex).setAxisRange(xmin,xmax,ymin,ymax); return }catch(Throwable t){}
  try{ def pad=canvas.getPad(padIndex); def af=pad.getAxisFrame()
    try{ af.getAxisX().setRange(xmin,xmax)}catch(Throwable t1){}
    try{ af.getAxisY().setRange(ymin,ymax)}catch(Throwable t2){}
  }catch(Throwable tt){}
}

// ----------------------- Data structs --------------------------------
class RecP{ int pid; float px,py,pz,vx,vy,vz,vt; byte charge; float beta,chi2pid; short status }
class KFRec{ int idx=-1; float px,py,chi2; int nhits; double phiDeg=Double.NaN; int trackid=-1 }
final class SLW{
  final int sl,l,w
  SLW(int a,int b,int c){ sl=a; l=b; w=c }
  int hashCode(){ return ((sl*1315423911)^(l*2654435761))^w }
  boolean equals(Object o){ if(!(o instanceof SLW)) return false; SLW x=(SLW)o; return x.sl==sl && x.l==l && x.w==w }
}
final class SL{
  final int sl,l
  SL(int a,int b){ sl=a; l=b }
  int hashCode(){ return (sl*1315423911)^l }
  boolean equals(Object o){ if(!(o instanceof SL)) return false; SL x=(SL)o; return x.sl==sl && x.l==l }
}
final class WireKey{
  final int s,Lraw,c
  WireKey(int s,int Lraw,int c){ this.s=s; this.Lraw=Lraw; this.c=c }
  int hashCode(){ return (s*73856093)^(Lraw*19349663)^(c*83492791) }
  boolean equals(Object o){ if(!(o instanceof WireKey)) return false; WireKey k=(WireKey)o; return k.s==s && k.Lraw==Lraw && k.c==c }
  String toString(){ return String.format("S%d L%02d C%d",s,Lraw,c) }
}

// --------------------------- Readers --------------------------------
RecP getElectronREC(DataEvent ev){
  if(!ev.hasBank("REC::Particle")) return null
  DataBank b=ev.getBank("REC::Particle"); int best=-1
  for(int i=0;i<b.rows();i++){
    if(b.getInt("pid",i)!=11) continue
    float vz=b.getFloat("vz",i); short st=b.getShort("status",i)
    if(vz<VZ_MIN || vz>VZ_MAX) continue
    if(FD_ONLY && st>=0) continue
    best=i; break
  }
  if(best<0) return null
  RecP e=new RecP()
  e.pid=b.getInt("pid",best)
  e.px=b.getFloat("px",best); e.py=b.getFloat("py",best); e.pz=b.getFloat("pz",best)
  e.vx=b.getFloat("vx",best); e.vy=b.getFloat("vy",best); e.vz=b.getFloat("vz",best); e.vt=b.getFloat("vt",best)
  e.charge=b.getByte("charge",best); e.beta=b.getFloat("beta",best); e.chi2pid=b.getFloat("chi2pid",best)
  e.status=b.getShort("status",best)
  return e
}
double W_from_e(double Ebeam,RecP e){
  double Ee=Math.sqrt((double)e.px*e.px+(double)e.py*e.py+(double)e.pz*e.pz)
  double qx=-(double)e.px, qy=-(double)e.py, qz=Ebeam-(double)e.pz, q0=Ebeam-Ee
  double Eh=M_D+q0; double w2=Eh*Eh-(qx*qx+qy*qy+qz*qz)
  return (w2>0.0d)? Math.sqrt(w2): Double.NaN
}
KFRec bestKF_BackToBack(DataEvent ev,double phi_e){
  KFRec out=new KFRec(); if(!ev.hasBank("AHDC::kftrack")) return out
  DataBank k=ev.getBank("AHDC::kftrack"); double bestAbs=Double.POSITIVE_INFINITY
  for(int i=0;i<k.rows();i++){
    int nh=k.getInt("n_hits",i); if(nh<KF_NHITS_MIN) continue
    float chi2=k.getFloat("chi2",i); if(!Float.isNaN(chi2) && (double)chi2>KF_CHI2_MAX) continue
    float px=k.getFloat("px",i), py=k.getFloat("py",i); double pk=phiDeg(px,py)
    double dphi=Math.abs(dphi0to360(phi_e,pk)-180.0d)
    if(dphi<bestAbs){ bestAbs=dphi; out.idx=i; out.px=px; out.py=py; out.chi2=chi2; out.nhits=nh; out.phiDeg=pk; out.trackid=k.getInt("trackid",i) }
  }
  return out
}
KFRec bestKF_Heavy(DataEvent ev){
  KFRec out=new KFRec(); if(!ev.hasBank("AHDC::kftrack")) return out
  DataBank k=ev.getBank("AHDC::kftrack"); double best=-1.0
  for(int i=0;i<k.rows();i++){
    int nh=k.getInt("n_hits",i); if(nh<KF_NHITS_MIN) continue
    float chi2=k.getFloat("chi2",i); if(!Float.isNaN(chi2) && chi2>KF_CHI2_MAX) continue
    int sadc=0; try{ sadc=k.getInt("sum_adc",i) }catch(Exception ignore){}
    double score=(nh>0? (double)sadc/nh : -1.0)
    if(score>best){ best=score; out.idx=i; out.nhits=nh; out.chi2=chi2; out.px=k.getFloat("px",i); out.py=k.getFloat("py",i)
      out.phiDeg=phiDeg(out.px,out.py); out.trackid=k.getInt("trackid",i) }
  }
  return out
}

// ------------------------ WF gating ---------------------------------
Set<String> wfExplicitGood(DataEvent ev){
  HashSet<String> good=new HashSet<String>(); if(!ev.hasBank("AHDC::wf")) return good
  DataBank w=ev.getBank("AHDC::wf")
  for(int i=0;i<w.rows();i++){
    int flag; try{ flag=w.getInt("flag",i) }catch(Exception e){ continue }
    if(flag!=0 && flag!=1) continue
    int s,LencVal,c
    try{ int sl=w.getInt("superlayer",i), l=w.getInt("layer",i); s=w.getInt("sector",i); c=w.getInt("component",i); LencVal=Lenc(sl,l) }
    catch(Exception e){ try{ s=w.getInt("sector",i); c=w.getInt("component",i); int Lraw=w.getInt("layer",i); LencVal=Lraw }catch(Exception ee){ continue } }
    good.add(s+"#"+LencVal+"#"+c)
  }
  return good
}
boolean wfPassForAdcRow(DataBank a,int i,Set<String> wfGoodSet){
  int s=a.getInt("sector",i), Lraw=a.getInt("layer",i), c=a.getInt("component",i)
  if(wfGoodSet.contains(s+"#"+Lraw+"#"+c)) return true
  int wft=Integer.MIN_VALUE; try{ wft=a.getInt("wfType",i) }catch(Exception ignore){}
  if(wft!=Integer.MIN_VALUE && wft>2) return false
  Double tot=null; try{ tot=(double)a.getFloat("timeOverThreshold",i) }catch(Exception ignore){}
  if(tot!=null && (tot<250.0 || tot>1200.0)) return false
  return true
}

// ---------------------- Track association ---------------------------
class AssocSets{ Set<SLW> slw=new HashSet<SLW>(); Set<SL> sl=new HashSet<SL>() }
AssocSets buildAssocSetsForTrackId(DataEvent ev,int wantedId){
  AssocSets as=new AssocSets(); if(!ev.hasBank("AHDC::hits")) return as
  DataBank h=ev.getBank("AHDC::hits")
  for(int i=0;i<h.rows();i++){
    int tid=h.getInt("trackid",i); if(tid!=wantedId) continue
    int sl=(h.getByte("superlayer",i)&0xFF), l=(h.getByte("layer",i)&0xFF), w=h.getInt("wire",i)
    as.slw.add(new SLW(sl,l,w)); as.sl.add(new SL(sl,l))
  }
  return as
}

// ----------------------- α from control window ----------------------
class AlphaCtrlRes{ double alphaLSQ, alphaRatio; int nBins; double Ssum,Bsum }
AlphaCtrlRes alphaFromControl(H1F sig,H1F bkg,double amin,double amax){
  def ax=sig.getAxis(); int n=ax.getNBins(); double xmin=ax.min(),xmax=ax.max(),w=(xmax-xmin)/n
  double SB=0,BB=0; int used=0; double Ssum=0,Bsum=0
  for(int b=0;b<n;b++){
    double xL=xmin+b*w, xR=xL+w; if(xR<=amin || xL>=amax) continue
    double S=sig.getBinContent(b), B=bkg.getBinContent(b); if(B<=0 && S<=0) continue
    double wt=1.0/Math.max(1.0,S+B); SB+=wt*S*B; BB+=wt*B*B; used++; Ssum+=S; Bsum+=B
  }
  AlphaCtrlRes out=new AlphaCtrlRes(); out.alphaLSQ=(BB>0? SB/BB:0.0); out.alphaRatio=(Bsum>0? Ssum/Bsum:0.0)
  out.nBins=used; out.Ssum=Ssum; out.Bsum=Bsum; return out
}

// ---------- MPV error (width/sqrt(N)) -------
double estimateMPVError(H1F h,double mpv,double xmin,double xmax){
  if(h==null||Double.isNaN(mpv)||mpv<=0) return 0.0
  def ax=h.getAxis(); int n=ax.getNBins(); double sumW=0,sumVar=0
  for(int b=0;b<n;b++){
    double x=ax.getBinCenter(b); if(x<xmin||x>xmax) continue
    double c=h.getBinContent(b); if(c<=0) continue
    sumW+=c; double dx=x-mpv; sumVar+=c*dx*dx
  }
  if(sumW<=1.0) return 0.0
  double sigmaWidth=Math.sqrt(sumVar/sumW)
  return sigmaWidth/Math.sqrt(sumW)
}

// -------------------- Landau fits --------------------
class FitResultGlobal{
  boolean ok=false; double amp,mpv,sigma,mpvErr,chi2=Double.NaN; int ndf=0
  F1D func=null  // keep function so we can draw it and show stats box
}
FitResultGlobal fitGlobalLandau(H1F h,double xmin,double xmax){
  FitResultGlobal out=new FitResultGlobal(); if(h.integral()<200.0) return out
  int binMax=h.getMaximumBin(); double mpv0=h.getAxis().getBinCenter(binMax)
  double amp0=Math.max(1.0,h.getBinContent(binMax)); double sig0=Math.max(50.0,(xmax-xmin)/10.0)
  if(mpv0<xmin||mpv0>xmax) mpv0=0.5*(xmin+xmax)
  F1D f=new F1D("f_global","[A]*landau(x,[MPV],[SIG])",xmin,xmax)
  f.setParameter(0,amp0); f.setParameter(1,mpv0); f.setParameter(2,sig0)
  try{ DataFitter.fit(f,h,"Q") }catch(Exception ex){ return out }
  out.amp=f.getParameter(0); out.mpv=f.getParameter(1); out.sigma=f.getParameter(2)
  out.mpvErr=estimateMPVError(h,out.mpv,xmin,xmax)
  // manual chi2/ndf
  def ax=h.getAxis(); int nBins=ax.getNBins(); int used=0; double chisq=0
  for(int b=0;b<nBins;b++){
    double x=ax.getBinCenter(b); if(x<xmin||x>xmax) continue
    double y=h.getBinContent(b); if(y<=0) continue
    double yfit=f.evaluate(x); double err=Math.sqrt(y); if(err<=0) err=1.0
    double d=(y-yfit)/err; chisq+=d*d; used++
  }
  int nPars=f.getNPars(); int ndf=used-nPars; if(ndf>0){ out.chi2=chisq; out.ndf=ndf }
  // style + enable stats box
  f.setLineColor(3); f.setLineWidth(3); f.setOptStat(1111)
  out.func=f
  out.ok=(out.mpv>0 && !Double.isNaN(out.mpv)); return out
}

class FitResult1D{
  boolean ok=false; double amp,mpv,sigma,ampErr,mpvErr,sigErr,chi2=Double.NaN; int ndf=0
  F1D func=null        // fitted function to draw stats box
  H1F fitCurve=null    // optional sampled curve (kept for compatibility)
}
FitResult1D fitLandau1D(H1F h,double xmin,double xmax,String tag,int nSampleBins){
  FitResult1D out=new FitResult1D(); if(h==null||h.integral()<80.0) return out
  int binMax=h.getMaximumBin(); double mpv0=h.getAxis().getBinCenter(binMax)
  double amp0=Math.max(1.0,h.getBinContent(binMax)); double sig0=Math.max(30.0,(xmax-xmin)/12.0)
  F1D f=new F1D("f_"+tag,"[A]*landau(x,[MPV],[SIG])",xmin,xmax)
  f.setParameter(0,amp0); f.setParameter(1,mpv0); f.setParameter(2,sig0)
  try{ DataFitter.fit(f,h,"Q") }catch(Throwable t){ return out }
  out.amp=f.getParameter(0); out.mpv=f.getParameter(1); out.sigma=f.getParameter(2)
  out.ampErr=0.0; out.sigErr=0.0; out.mpvErr=estimateMPVError(h,out.mpv,xmin,xmax)
  // keep sampled curve (unchanged)
  H1F fc=new H1F("fitcurve_"+tag,"",nSampleBins,h.getAxis().min(),h.getAxis().max())
  def ax2=fc.getAxis(); int nb=ax2.getNBins()
  for(int b=0;b<nb;b++){
    double x=ax2.getBinCenter(b)
    double y=(x>=xmin&&x<=xmax)? f.evaluate(x):0.0
    fc.setBinContent(b,y)
  }
  fc.setLineColor(3); out.fitCurve=fc
  // style + enable stats box on the function
  f.setLineColor(3); f.setLineWidth(3); f.setOptStat(1111)
  out.func=f

  // chi2/ndf
  def ax=h.getAxis(); int used=0; double chisq=0
  for(int b=0;b<ax.getNBins();b++){
    double x=ax.getBinCenter(b); if(x<xmin||x>xmax) continue
    double y=h.getBinContent(b); if(y<=0) continue
    double yfit=f.evaluate(x); double err=Math.sqrt(y); if(err<=0) err=1.0
    double d=(y-yfit)/err; chisq+=d*d; used++
  }
  int nPars=f.getNPars(); int ndf=used-nPars; if(ndf>0){ out.chi2=chisq; out.ndf=ndf }

  out.ok=(out.mpv>0 && !Double.isNaN(out.mpv)); return out
}

// -------------------- Maps & globals for per-wire --------------------
@Field Map<WireKey,PairH>  histMap    = new LinkedHashMap<>()
@Field Map<WireKey,H1F>    subGainMap = new LinkedHashMap<>()
@Field Map<WireKey,Double> alphaMap   = new LinkedHashMap<>()

final class PairH{
  final H1F sig,bkg,sub
  PairH(String tag,String titleBase,int nb,double lo,double hi){
    sig=new H1F("sig_"+tag,     titleBase+" (SIG, p_{T} slice);ADC;Counts",nb,lo,hi)
    bkg=new H1F("bkg_"+tag,     titleBase+" (BKG, p_{T} slice);ADC;Counts",nb,lo,hi)
    sub=new H1F("sub_"+tag,     titleBase+" (SUB=SIG−α·BKG);ADC;Counts",nb,lo,hi)
    sig.setLineColor(4)   // blue for SIG
    bkg.setLineColor(2)   // red for BKG
    sub.setLineColor(4)   // blue for SUB (pre)
  }
}

// ΣADC sums for global reference
@Field H1F   SUM_SIG   = new H1F("sum_sig","Per-wire ADC (SIG, all wires);ADC;Counts",ADC_NBINS,0.0,ADC_MAX)
@Field H1F   SUM_BKG   = new H1F("sum_bkg","Per-wire ADC (BKG, all wires);ADC;Counts",ADC_NBINS,0.0,ADC_MAX)
@Field H1F   SUM_SUB   = new H1F("sum_sub","Per-wire ADC (SUB=SIG−α·BKG, all wires);ADC;Counts",ADC_NBINS,0.0,ADC_MAX)
@Field H1F   SUM_SUB_FIT=null
void styleSum(){ SUM_SIG.setLineColor(1); SUM_BKG.setLineColor(2); SUM_SUB.setLineColor(4) }
styleSum()

PairH getPair(WireKey k){
  PairH p=histMap.get(k)
  if(p==null){
    String tag=k.toString().replace(' ','_'), ttl="ADC — "+k.toString()+" (p_{T} slice)"
    p=new PairH(tag,ttl,ADC_NBINS,0.0,ADC_MAX); histMap.put(k,p)
  }
  return p
}

// -------------------- 2D pT–ADC (SUB-only) containers ---------------
@Field H2F H2_PRE_ALL  = new H2F("h2_pt_adc_pre_all", "p_{T} vs ADC (SUB events, RAW);p_{T} (GeV);ADC", PT2_NBINS, PT2_MIN, PT2_MAX, ADC2_NBINS, ADC2_MIN, ADC2_MAX)
@Field H2F H2_POST_ALL = new H2F("h2_pt_adc_post_all","p_{T} vs ADC_{corr} (SUB events);p_{T} (GeV);ADC_{corr}", PT2_NBINS, PT2_MIN, PT2_MAX, ADC2_NBINS, 0.0, ADC2_MAX)
@Field H2F H2_PRE_L    = new H2F("h2_pt_adc_pre_L",   "LEFT: p_{T} vs ADC (SUB events);p_{T} (GeV);ADC", PT2_NBINS, PT2_MIN, PT2_MAX, ADC2_NBINS, ADC2_MIN, ADC2_MAX)
@Field H2F H2_PRE_R    = new H2F("h2_pt_adc_pre_R",   "RIGHT: p_{T} vs ADC (SUB events);p_{T} (GeV);ADC", PT2_NBINS, PT2_MIN, PT2_MAX, ADC2_NBINS, ADC2_MIN, ADC2_MAX)
@Field H2F H2_POST_L   = new H2F("h2_pt_adc_post_L",  "LEFT: p_{T} vs ADC_{corr} (SUB events);p_{T} (GeV);ADC_{corr}", PT2_NBINS, PT2_MIN, PT2_MAX, ADC2_NBINS, 0.0, ADC2_MAX)
@Field H2F H2_POST_R   = new H2F("h2_pt_adc_post_R",  "RIGHT: p_{T} vs ADC_{corr} (SUB events);p_{T} (GeV);ADC_{corr}", PT2_NBINS, PT2_MIN, PT2_MAX, ADC2_NBINS, 0.0, ADC2_MAX)

// ----------------------------- CLI ----------------------------------
OptionStore opt=new OptionStore("pd7plus")
opt.addCommand("process","Per-wire ADC (SUB) calibration with MPV fits; overlays; 2D SUB pre/post")
final def cli=opt.getOptionParser("process")
cli.addOption("-nevent",""); cli.addOption("-beam","")
cli.addOption("-w2min",""); cli.addOption("-w2max","")
cli.addOption("-dphiHalf",""); cli.addOption("-banana",""); cli.addOption("-mode_valid","")
cli.addOption("-vzmin",""); cli.addOption("-vzmax",""); cli.addOption("-fdonly","")
cli.addOption("-ptmin",""); cli.addOption("-ptmax","")
cli.addOption("-ctrlLo",""); cli.addOption("-ctrlHi","")
cli.addOption("-trackid","")
// Optional 2D QA range (kept)
cli.addOption("-pt2min",""); cli.addOption("-pt2max",""); cli.addOption("-pt2bins","")
cli.addOption("-adc2max","")
opt.parse(args)
if(opt.getCommand()!="process"){ System.err.println("Usage: run-groovy gain_MPV_proton_deutron_final_test4_fitparam_box.groovy process [opts] files.hipo ..."); System.exit(1) }

// collect inputs (glob aware)
List<String> expandGlob(String pat){
  ArrayList<String> out=new ArrayList<String>(); if(pat==null) return out
  File f=new File(pat)
  if(f.exists()&&f.isFile()){ out.add(f.getPath()); return out }
  if(pat.indexOf('*')>=0 || pat.indexOf('?')>=0){
    File parent=f.getParentFile(); if(parent==null) parent=new File(".")
    String rx="\\Q"+f.getName().replace("?", "\\E.\\Q").replace("*","\\E.*\\Q")+"\\E"
    def re=java.util.regex.Pattern.compile(rx)
    File[] list=parent.listFiles(); if(list!=null){ for(File ff: list){ if(ff.isFile() && re.matcher(ff.getName()).matches()) out.add(ff.getPath()) } }
  }
  return out
}
ArrayList<String> rawInputs=new ArrayList<String>(); for(String s: cli.getInputList()) rawInputs.add(s)
ArrayList<String> files=new ArrayList<String>(); for(String s: rawInputs){
  if(s==null) continue
  if(s.toLowerCase(Locale.ROOT).endsWith(".hipo")) files.add(s)
  else if(s.indexOf('*')>=0 || s.indexOf('?')>=0) files.addAll(expandGlob(s))
}
LinkedHashSet<String> uniq=new LinkedHashSet<String>(files); files.clear(); files.addAll(uniq)
Iterator<String> itf=files.iterator(); while(itf.hasNext()){ String f=itf.next(); if(!(new File(f).isFile())) itf.remove() }
if(files.isEmpty()){ System.err.println("No .hipo inputs."); System.exit(1) }

// parse options (unchanged)
String v
try{ v=cli.getOption("-nevent")?.getValue(); if(v!=null) MAXEV=Long.parseLong(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-beam")?.getValue();  if(v!=null) EBEAM=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-w2min")?.getValue(); if(v!=null) W2_MIN=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-w2max")?.getValue(); if(v!=null) W2_MAX=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-dphiHalf")?.getValue(); if(v!=null) DPHI_HALF=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-banana")?.getValue(); if(v!=null) BANANA_ON=Boolean.parseBoolean(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-mode_valid")?.getValue(); if(v!=null) BANANA_ON = v.trim().equalsIgnoreCase("banana") }catch(Exception ignore){}
try{ v=cli.getOption("-vzmin")?.getValue(); if(v!=null) VZ_MIN=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-vzmax")?.getValue(); if(v!=null) VZ_MAX=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-fdonly")?.getValue(); if(v!=null) FD_ONLY=Boolean.parseBoolean(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-ptmin")?.getValue(); if(v!=null) PT_SLICE_MIN=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-ptmax")?.getValue(); if(v!=null) PT_SLICE_MAX=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-ctrlLo")?.getValue(); if(v!=null) CTRL_LO=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-ctrlHi")?.getValue(); if(v!=null) CTRL_HI=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-pt2min")?.getValue(); if(v!=null) PT2_MIN=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-pt2max")?.getValue(); if(v!=null) PT2_MAX=Double.parseDouble(v.trim()) }catch(Exception ignore){}
try{ v=cli.getOption("-pt2bins")?.getValue(); if(v!=null) PT2_NBINS=Integer.parseInt(v.trim()) }catch(Exception ignore){}
try{
  v=cli.getOption("-adc2max")?.getValue();
  if(v!=null){
    ADC2_MAX=Double.parseDouble(v.trim())
    H2_PRE_ALL.getYAxis().setRange(ADC2_MIN,ADC2_MAX)
    H2_POST_ALL.getYAxis().setRange(0.0,ADC2_MAX)
    H2_PRE_L.getYAxis().setRange(ADC2_MIN,ADC2_MAX)
    H2_PRE_R.getYAxis().setRange(ADC2_MIN,ADC2_MAX)
    H2_POST_L.getYAxis().setRange(0.0,ADC2_MAX)
    H2_POST_R.getYAxis().setRange(0.0,ADC2_MAX)
  }
}catch(Exception ignore){}
Integer FORCE_TRACK_ID=null; try{ v=cli.getOption("-trackid")?.getValue(); if(v!=null) FORCE_TRACK_ID=Integer.valueOf(v.trim()) }catch(Exception ignore){}

// ------------------------------ PASS-1: build SIG/BKG ----------------
ProgressPrintout prog=new ProgressPrintout(); long seen=0L
for(String fn: files){
  HipoDataSource R=new HipoDataSource()
  try{ R.open(fn) }catch(Exception ex){ System.err.println("Open fail "+fn+" : "+ex); continue }
  while(R.hasEvent()){
    DataEvent ev; try{ ev=R.getNextEvent() }catch(Exception ex){ break }
    seen++
    RecP e=getElectronREC(ev); if(e==null){ prog.updateStatus(); if(MAXEV>0 && seen>=MAXEV) break; else continue }
    double pt=Math.hypot((double)e.px,(double)e.py)
    if(pt<PT_SLICE_MIN || pt>PT_SLICE_MAX){ prog.updateStatus(); if(MAXEV>0 && seen>=MAXEV) break; else continue }

    double phi_e=phiDeg(e.px,e.py)
    KFRec kf_back=bestKF_BackToBack(ev,phi_e), kf_heavy=bestKF_Heavy(ev)
    if(kf_back.idx<0 || kf_heavy.idx<0){ prog.updateStatus(); if(MAXEV>0 && seen>=MAXEV) break; else continue }

    int assocId=(FORCE_TRACK_ID!=null)? FORCE_TRACK_ID.intValue(): kf_heavy.trackid
    if(assocId<0){ prog.updateStatus(); if(MAXEV>0 && seen>=MAXEV) break; else continue }
    AssocSets AS=buildAssocSetsForTrackId(ev,assocId)

    double W=W_from_e(EBEAM,e); double W2=(Double.isNaN(W)? Double.NaN: W*W)
    boolean okW=(!Double.isNaN(W2) && between(W2,W2_MIN,W2_MAX))
    boolean okDP=(Math.abs(dphi0to360(phi_e,kf_back.phiDeg)-180.0d)<=DPHI_HALF)

    long sumTRK=0L
    if(ev.hasBank("AHDC::adc")){
      DataBank a0=ev.getBank("AHDC::adc")
      for(int i=0;i<a0.rows();i++){
        int A=0; try{ A=a0.getInt("ADC",i) }catch(Exception ignore){}
        if(A<=0) continue
        int Lraw=a0.getInt("layer",i), c=a0.getInt("component",i)
        int sl=Lraw/10, l=Lraw%10
        if(AS.slw.contains(new SLW(sl,l,c))) sumTRK+=(long)A
      }
    }
    try{ DataBank k=ev.getBank("AHDC::kftrack"); int sadc=k.getInt("sum_adc",kf_heavy.idx); if(sadc>0) sumTRK=(long)sadc }catch(Exception ignore){}
    boolean okBan=(!BANANA_ON)||inBanana(pt,sumTRK)
    boolean ELASTIC= okW && okDP && okBan

    if(!ev.hasBank("AHDC::adc")){ prog.updateStatus(); if(MAXEV>0 && seen>=MAXEV) break; else continue }
    DataBank a=ev.getBank("AHDC::adc")
    Set<String> wfGoodSet=wfExplicitGood(ev)

    for(int i=0;i<a.rows();i++){
      int A=0; try{ A=a.getInt("ADC",i) }catch(Exception ignore){}
      if(A<=0) continue; if(!wfPassForAdcRow(a,i,wfGoodSet)) continue
      int Lraw=a.getInt("layer",i), c=a.getInt("component",i); int sl=Lraw/10, l=Lraw%10; int s=a.getInt("sector",i)
      if(!AS.slw.contains(new SLW(sl,l,c))) continue
      double xadc=clampADC((double)A,ADC_MAX)
      WireKey wk=new WireKey(s,Lraw,c); PairH ph=getPair(wk)
      if(ELASTIC){ ph.sig.fill(xadc); SUM_SIG.fill(xadc) } else { ph.bkg.fill(xadc); SUM_BKG.fill(xadc) }
    }
    prog.updateStatus(); if(MAXEV>0 && seen>=MAXEV) break
  }
  try{ R.close() }catch(Exception ignore){}
}

// ----------------------- α per wire & SUB ----------------------------
AlphaCtrlRes arSUM=alphaFromControl(SUM_SIG,SUM_BKG,CTRL_LO,CTRL_HI)
double A_glob = arSUM.alphaLSQ>0 ? arSUM.alphaLSQ : (arSUM.alphaRatio>0? arSUM.alphaRatio : 0.0)
A_glob=clampAlpha(A_glob)

for(Map.Entry<WireKey,PairH> e: histMap.entrySet()){
  WireKey wk=e.getKey(); PairH ph=e.getValue()
  AlphaCtrlRes ar=alphaFromControl(ph.sig,ph.bkg,CTRL_LO,CTRL_HI)
  double aWire
  boolean goodStats=(ar.nBins>=CTRL_MIN_BINS && ar.Bsum>=CTRL_MIN_BKG_SUM)
  if(goodStats){ if(ar.alphaLSQ>0) aWire=ar.alphaLSQ; else if(ar.alphaRatio>0) aWire=ar.alphaRatio; else aWire=A_glob }
  else aWire=A_glob
  aWire=clampAlpha(aWire); alphaMap.put(wk,aWire)
  int n=ph.sub.getAxis().getNBins()
  for(int b=0;b<n;b++){
    double y=ph.sig.getBinContent(b)-aWire*ph.bkg.getBinContent(b)
    if(ZERO_FLOOR_SUB && y<0) y=0
    ph.sub.setBinContent(b,y)
  }
}
for(int b=0;b<SUM_SUB.getAxis().getNBins();b++){
  double y=SUM_SIG.getBinContent(b)-A_glob*SUM_BKG.getBinContent(b)
  if(ZERO_FLOOR_SUB && y<0) y=0
  SUM_SUB.setBinContent(b,y)
}

// ------------------------ Global Landau (REF) -----------------------
FitResultGlobal frSum=fitGlobalLandau(SUM_SUB,FIT_LO,FIT_HI)
double MPV_REF=Double.NaN, MPV_REF_ERR=0.0
if(frSum.ok){
  MPV_REF=frSum.mpv; MPV_REF_ERR=frSum.mpvErr
  // sampled copy (unchanged)
  SUM_SUB_FIT=new H1F("sum_sub_fit","",ADC_NBINS,0.0,ADC_MAX)
  def axS=SUM_SUB_FIT.getAxis(); int nbS=axS.getNBins()
  for(int b=0;b<nbS;b++){
    double x=axS.getBinCenter(b)
    double y=frSum.amp * FunctionFactory.landau(x,frSum.mpv,frSum.sigma)
    SUM_SUB_FIT.setBinContent(b,y)
  }
  SUM_SUB_FIT.setLineColor(3)
}else{
  System.out.println("Global SUM_SUB MPV: not fitted (too few entries)")
}

// --------------------- Per-wire Landau fits (pre/post), gains --------
class FitPack { FitResult1D pre; FitResult1D post; double g; double ge; }
Map<WireKey,FitPack> fitPackMap=new LinkedHashMap<>()

for(Map.Entry<WireKey,PairH> e : histMap.entrySet()){
  WireKey wk=e.getKey(); PairH ph=e.getValue()
  FitPack pack=new FitPack()
  pack.pre = (DO_PER_WIRE_FITS? fitLandau1D(ph.sub, FIT_LO, FIT_HI, "pre_"+wk.toString().replace(' ','_'), ADC_NBINS) : new FitResult1D())

  pack.g  = 1.0; pack.ge = 0.0
  if(pack.pre!=null && pack.pre.ok && MPV_REF>0){
    pack.g = MPV_REF / pack.pre.mpv
    double rel_ref =(MPV_REF_ERR>0? MPV_REF_ERR/MPV_REF:0.0)
    double rel_wire=(pack.pre.mpv>0 && pack.pre.mpvErr>0? (pack.pre.mpvErr/pack.pre.mpv):0.0)
    pack.ge = pack.g*Math.sqrt(rel_ref*rel_ref + rel_wire*rel_wire)
  }

  // build per-wire SUB gain-corrected histogram by rebinning SUB counts
  H1F hSub=ph.sub
  H1F hCorr=new H1F("sub_gain_"+wk.toString().replace(' ','_'),
                    "ADC_{corr} — "+wk.toString()+" (SUB gain-corr);ADC_{corr};Counts",
                    ADC_NBINS,0.0,ADC_MAX)
  def axOld=hSub.getAxis(); int nOld=axOld.getNBins()
  double xMinOld=axOld.min(), xMaxOld=axOld.max(), bwOld=(xMaxOld-xMinOld)/nOld
  def axNew=hCorr.getAxis(); int nNew=axNew.getNBins(); double xMinNew=axNew.min(), xMaxNew=axNew.max(), bwNew=(xMaxNew-xMinNew)/nNew
  for(int b=0;b<nOld;b++){
    double c=hSub.getBinContent(b); if(c<=0) continue
    double x=axOld.getBinCenter(b); double xCorr=x*(pack.g>0?pack.g:1.0)
    xCorr=clampADC(xCorr, ADC_MAX)
    if(xCorr<xMinNew || xCorr>=xMaxNew) continue
    int ibNew=(int)Math.floor((xCorr - xMinNew)/bwNew)
    if(ibNew>=0 && ibNew<nNew) hCorr.setBinContent(ibNew, hCorr.getBinContent(ibNew)+c)
  }
  hCorr.setLineColor(6) // magenta for post
  subGainMap.put(wk, hCorr)

  // post-gain MPV fit
  pack.post = (DO_PER_WIRE_FITS? fitLandau1D(hCorr, FIT_LO, FIT_HI, "post_"+wk.toString().replace(' ','_'), ADC_NBINS) : new FitResult1D())

  fitPackMap.put(wk, pack)
}

// ----------------------------- GUI #1 (QA) --------------------------
ArrayList<WireKey> allKeys=new ArrayList<WireKey>(histMap.keySet())
Collections.sort(allKeys,new Comparator<WireKey>(){ int compare(WireKey a,WireKey b){ if(a.s!=b.s) return a.s-b.s; if(a.Lraw!=b.Lraw) return a.Lraw-b.Lraw; return a.c-b.c } })
int total=allKeys.size(); int pages=(int)Math.ceil(total/(double)PAGE_SIZE())

ArrayList<String> tabs=new ArrayList<String>()
for(int p=1;p<=pages;p++){
  tabs.add(String.format("SUB [p%d/%d]",p,pages))
  tabs.add(String.format("SUB gain [p%d/%d]",p,pages))
}
tabs.add("SUM 1D & REF")
tabs.add("Wire MPV & Gain")

EmbeddedCanvasTabbed canv=new EmbeddedCanvasTabbed(tabs.toArray(new String[0]))

def drawSubPage={ String name, boolean post, int pageIdx ->
  def cx=canv.getCanvas(name); cx.divide(DRAW_COLS,DRAW_ROWS)
  int start=(pageIdx-1)*PAGE_SIZE(), end=Math.min(start+PAGE_SIZE(),total), pad=0
  for(int i=start;i<end;i++){
    WireKey wk=allKeys.get(i); cx.cd(pad)
    H1F h = post ? subGainMap.get(wk) : histMap.get(wk).sub
    if(h==null){ pad++; continue }
    h.setLineColor(post?6:4)
    cx.draw(h)
    if(DO_PER_WIRE_FITS && DRAW_GREEN_FITS){
      FitPack pack=fitPackMap.get(wk)
      def fr = post ? (pack!=null? pack.post: null) : (pack!=null? pack.pre: null)
      if(fr!=null && fr.ok){
        if(fr.func!=null){
          cx.draw(fr.func,"same")   // <<< draws the curve AND GROOT stats box
        }else if(fr.fitCurve!=null){
          cx.draw(fr.fitCurve,"same") // fallback
        }
      }
    }
    YSym yr=symmetricY(h,1.15); double xmin=h.getAxis().min(), xmax=h.getAxis().max()
    setPadRangesSafe(cx,pad,xmin,xmax,yr.ymin,yr.ymax); pad++
  }
}
for(int p=1;p<=pages;p++){
  drawSubPage(String.format("SUB [p%d/%d]",p,pages),false,p)
  drawSubPage(String.format("SUB gain [p%d/%d]",p,pages),true,p)
}

// SUM + REF
def csum=canv.getCanvas("SUM 1D & REF"); csum.divide(1,1); csum.cd(0); 
csum.draw(SUM_SIG); csum.draw(SUM_BKG); csum.draw(SUM_SUB)
if(SUM_SUB_FIT!=null) csum.draw(SUM_SUB_FIT,"same")
if(frSum!=null && frSum.ok && frSum.func!=null) csum.draw(frSum.func,"same")  // <<< stats box for global fit

// Graphs
GraphErrors gPre=new GraphErrors("MPV_pre_fit_vs_wire"), gPost=new GraphErrors("MPV_post_fit_vs_wire"), gGain=new GraphErrors("Gain_vs_wire")
int idx=0
for(WireKey wk: allKeys){
  double x=(double)idx++
  FitPack pack=fitPackMap.get(wk)
  if(pack!=null){
    if(pack.pre!=null && pack.pre.ok)  gPre.addPoint(x,pack.pre.mpv, 0.0,(pack.pre.mpvErr>0? pack.pre.mpvErr:0.0))
    if(pack.post!=null && pack.post.ok) gPost.addPoint(x,pack.post.mpv,0.0,(pack.post.mpvErr>0? pack.post.mpvErr:0.0))
    gGain.addPoint(x,pack.g,0.0,(pack.ge>0? pack.ge:0.0))
  }
}
gPre.setTitle("MPV (pre, FIT) vs wire;wire index;MPV_{pre}^{fit} (ADC)")
gPost.setTitle("MPV (post, FIT) vs wire;wire index;MPV_{post}^{fit} (ADC)")
gGain.setTitle("Gain vs wire (FIT method);wire index;gain")
[gPre,gPost,gGain].each{ gr-> gr.setMarkerStyle(2); gr.setMarkerSize(4) }
gPre.setMarkerColor(1); gPre.setLineColor(1)
gPost.setMarkerColor(2); gPost.setLineColor(2)
gGain.setMarkerColor(3); gGain.setLineColor(3)
def cwg=canv.getCanvas("Wire MPV & Gain"); cwg.divide(2,2); cwg.cd(0); cwg.draw(gPre); cwg.cd(1); cwg.draw(gPost); cwg.cd(2); cwg.draw(gGain)
try{ cwg.getPad(2).getAxisFrame().getAxisY().setRange(0.7,1.3) }catch(Exception ignore){}

// bring up GUI #1
try{
  JFrame f1=new JFrame("Per-wire calibration — SUB / SUB gain / REF / MPV & Gain (with fit stats)")
  f1.setSize(1600, 900)
  f1.add(canv)
  f1.setVisible(true)
  f1.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
}catch(Exception ignore){}

// ============================ PASS-3: global pT–ADC 2D for SUB events ============================
boolean HAS_ORDER=false
ProgressPrintout prog3=new ProgressPrintout(); long seen3=0L
for(String fn: files){
  HipoDataSource R=new HipoDataSource()
  try{ R.open(fn) }catch(Exception ex){ System.err.println("Open fail "+fn+" : "+ex); continue }
  while(R.hasEvent()){
    DataEvent ev; try{ ev=R.getNextEvent() }catch(Exception ex){ break }
    seen3++
    RecP eRec=getElectronREC(ev); if(eRec==null){ prog3.updateStatus(); continue }
    double pt=Math.hypot((double)eRec.px,(double)eRec.py)
    double phi_e=phiDeg(eRec.px,eRec.py)

    KFRec kb=bestKF_BackToBack(ev,phi_e), kh=bestKF_Heavy(ev)
    if(kb.idx<0 || kh.idx<0){ prog3.updateStatus(); continue }
    int assocId=(FORCE_TRACK_ID!=null)? FORCE_TRACK_ID.intValue(): kh.trackid; if(assocId<0){ prog3.updateStatus(); continue }
    AssocSets AS=buildAssocSetsForTrackId(ev,assocId)

    double W=W_from_e(EBEAM,eRec); double W2=(Double.isNaN(W)? Double.NaN: W*W)
    boolean okW=(!Double.isNaN(W2) && between(W2,W2_MIN,W2_MAX))
    boolean okDP=(Math.abs(dphi0to360(phi_e,kb.phiDeg)-180.0d)<=DPHI_HALF)

    long sumTRK=0L
    if(ev.hasBank("AHDC::adc")){
      DataBank a0=ev.getBank("AHDC::adc")
      for(int i=0;i<a0.rows();i++){
        int A=0; try{ A=a0.getInt("ADC",i) }catch(Exception ignore){}
        if(A<=0) continue
        int Lraw=a0.getInt("layer",i), c=a0.getInt("component",i)
        int sl=Lraw/10, l=Lraw%10
        if(AS.slw.contains(new SLW(sl,l,c))) sumTRK+=(long)A
      }
    }
    try{ DataBank k=ev.getBank("AHDC::kftrack"); int sadc=k.getInt("sum_adc",kh.idx); if(sadc>0) sumTRK=(long)sadc }catch(Exception ignore){}
    boolean okBan=(!BANANA_ON)||inBanana(pt,sumTRK)
    boolean ELASTIC = okW && okDP && okBan
    if(!ELASTIC){ prog3.updateStatus(); continue }
    if(!ev.hasBank("AHDC::adc")){ prog3.updateStatus(); continue }
    DataBank a=ev.getBank("AHDC::adc")
    Set<String> wfGoodSet=wfExplicitGood(ev)

    for(int i=0;i<a.rows();i++){
      int A=0; try{ A=a.getInt("ADC",i) }catch(Exception ignore){}
      if(A<=0) continue; if(!wfPassForAdcRow(a,i,wfGoodSet)) continue
      int Lraw=a.getInt("layer",i), c=a.getInt("component",i)
      int sl=Lraw/10, l=Lraw%10; int s=a.getInt("sector",i)
      if(!AS.slw.contains(new SLW(sl,l,c))) continue

      double adcRaw=clampADC((double)A,ADC_MAX)
      double g=1.0; WireKey wk=new WireKey(s,Lraw,c); if(fitPackMap.containsKey(wk)){ FitPack pk=fitPackMap.get(wk); if(pk!=null && pk.g>0) g=pk.g }
      double adcCorr=clampADC(adcRaw*g, ADC_MAX)

      H2_PRE_ALL.fill(pt, adcRaw); H2_POST_ALL.fill(pt, adcCorr)

      // Optional left/right (order) split if present
      int ord=-1
      try{
        try{ ord=a.getByte("order",i) }catch(Exception eB){ ord = a.getInt("order",i) }
        HAS_ORDER=true
      }catch(Exception ignoreOrder){}
      if(HAS_ORDER){
        if(ord==0){ H2_PRE_L.fill(pt,adcRaw); H2_POST_L.fill(pt,adcCorr) }
        else if(ord==1){ H2_PRE_R.fill(pt,adcRaw); H2_POST_R.fill(pt,adcCorr) }
      }
    }
    prog3.updateStatus()
  }
  try{ R.close() }catch(Exception ignore){}
}

// ----------------------------- GUI #2 (overlays & 2D) -----------------
ArrayList<String> tabs2=new ArrayList<String>()
for(int p=1;p<=pages;p++) tabs2.add(String.format("Overlay pre/post [p%d/%d]",p,pages))
tabs2.add("pT–ADC (pre vs post)")
if(HAS_ORDER) tabs2.add("pT–ADC (L/R × pre/post)")

EmbeddedCanvasTabbed canvExtra=new EmbeddedCanvasTabbed(tabs2.toArray(new String[0]))

// Overlay pages
def drawOverlayPage={ String name, int pageIdx ->
  def cx=canvExtra.getCanvas(name); cx.divide(DRAW_COLS,DRAW_ROWS)
  int start=(pageIdx-1)*PAGE_SIZE(), end=Math.min(start+PAGE_SIZE(),total), pad=0
  for(int i=0;i<end-start;i++){
    WireKey wk=allKeys.get(start+i); cx.cd(pad)
    H1F pre=histMap.get(wk).sub
    H1F post=subGainMap.get(wk)
    if(pre!=null){ pre.setLineColor(4); cx.draw(pre) } // blue
    if(post!=null){ post.setLineColor(6); cx.draw(post,"same") } // magenta
    // also draw fit curves (with stats boxes) if desired on overlay:
    FitPack pack=fitPackMap.get(wk)
    if(pack!=null){
      if(pack.pre!=null && pack.pre.ok && pack.pre.func!=null) cx.draw(pack.pre.func,"same")
      if(pack.post!=null && pack.post.ok && pack.post.func!=null) cx.draw(pack.post.func,"same")
    }
    YSym ypre=(pre!=null? symmetricY(pre,1.15): new YSym(0,1))
    YSym ypost=(post!=null? symmetricY(post,1.15): new YSym(0,1))
    double ymin=Math.min(ypre.ymin, ypost.ymin), ymax=Math.max(ypre.ymax, ypost.ymax)
    double xmin=(pre!=null? pre.getAxis().min(): 0.0), xmax=(pre!=null? pre.getAxis().max(): ADC_MAX)
    setPadRangesSafe(cx,pad,xmin,xmax,ymin,ymax)
    pad++
  }
}
for(int p=1;p<=pages;p++) drawOverlayPage(String.format("Overlay pre/post [p%d/%d]",p,pages),p)

// 1×2: pT–ADC pre vs post
def c2d=canvExtra.getCanvas("pT–ADC (pre vs post)"); c2d.divide(2,1); c2d.cd(0); c2d.draw(H2_PRE_ALL); c2d.cd(1); c2d.draw(H2_POST_ALL)

// 2×2: L/R × pre/post (if order exists)
if(HAS_ORDER){
  def c2dlr=canvExtra.getCanvas("pT–ADC (L/R × pre/post)"); c2dlr.divide(2,2)
  c2dlr.cd(0); c2dlr.draw(H2_PRE_L)
  c2dlr.cd(1); c2dlr.draw(H2_PRE_R)
  c2dlr.cd(2); c2dlr.draw(H2_POST_L)
  c2dlr.cd(3); c2dlr.draw(H2_POST_R)
}

// bring up GUI #2
try{
  JFrame f2=new JFrame("Per-wire overlays & pT–ADC 2D (SUB only)")
  f2.setSize(1600, 900)
  f2.add(canvExtra)
  f2.setVisible(true)
  f2.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
}catch(Exception ignore){}
