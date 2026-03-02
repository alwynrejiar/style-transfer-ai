# @title Copy To Use Remotely
import json,time,requests as R
H={"ngrok-skip-browser-warning":"1","Content-Type":"application/json"}
E,P,G,X=R.exceptions,R.post,R.get,SystemExit
def main():
 d="https://myollamaapi2000.share.zrok.io";U=input(f"URL [{d}]: ").strip().rstrip("/") or d
 try:r=G(U,headers=H,timeout=10);s=r.status_code not in(200,404) and print(f"[WARN] {r.status_code}\n{r.text[:300]}")
 except E.ConnectionError:print(f"[!] {U} unreachable");raise X(1)
 try:r=G(f"{U}/api/tags",headers=H,timeout=15);r.status_code!=200 and(print(f"[!] {r.status_code}: {r.text[:400]}"),exit(1));M=[m["name"]for m in r.json().get("models",[])]
 except E.ConnectionError:print(f"[!] no connect");raise X(1)
 if not M:print("[!] no models");raise X(1)
 for i,n in enumerate(M,1):print(f" [{i}] {n}")
 m=None
 while not m:
  c=input("Model: ").strip()
  if c.isdigit()and 0<int(c)<=len(M):m=M[int(c)-1]
  elif c in M:m=c
 print(f"\n── {m} ──\n");h=[]
 while True:
  try:u=input("You: ").strip()
  except:print("\nBye!");break
  if u.lower()in{"quit","exit","q"}:break
  if not u:continue
  h+=[{"role":"user","content":u}];print("AI: ",end="",flush=True);a=""
  for t in range(1,5):
   try:
    with P(f"{U}/api/chat",headers=H,json={"model":m,"messages":h,"stream":True},stream=True,timeout=300)as r:
     if r.status_code==504:time.sleep(15*t);continue
     if not r.ok:print(f"\n[!]{r.status_code}:{r.text[:300]}");h.pop();break
     for l in r.iter_lines():
      if l:
       try:c=json.loads(l);d=c.get("message",{}).get("content","");print(d,end="",flush=True);a+=d
       except:0
       if c.get("done"):break
     break
   except E.Timeout:time.sleep(15*t)
   except E.RequestException as e:print(f"\n[!]{e}");h.pop();break
  else:print("\n[!]failed");h.pop()
  print();h+=[{"role":"assistant","content":a}]
if __name__=="__main__":main()
