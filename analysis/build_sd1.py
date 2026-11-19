import json, csv
C="/home/czx/2024/BandgapFusion_Consolidated/data/confirmation/"
recs=[x for x in json.load(open(C+"cohort.json"))["records"] if x.get("cohort")=="independent"]
enc={r["measurement_id"]:r for r in json.load(open(C+"encoded_features.json"))}
src="/data/czx/GapScope_Revision_20261113/polymorph/Supplementary_Data_1_GapScope_experimental_bandgaps.csv"
rows=list(csv.DictReader(open(src,encoding="utf-8-sig"))); cols=list(rows[0].keys())
cols=["record_group"]+cols
for r in rows: r["record_group"]="original_collection"
nz=lambda v: "" if v in (None,"None","nan") else str(v)
new=[]
for x in recs:
    e=enc.get(x["measurement_id"],{}); sgi=e.get("space_group")
    sg=nz(sgi)
    doi=x["source_group_id"][4:] if x["source_group_id"].startswith("doi:") else ""
    new.append({c:"" for c in cols}|dict(record_group="additional_sources",record_id="additional:"+x["measurement_id"],formula=x["formula"],formula_as_reported=x["formula"],
      composition_key=x["composition_key"],bandgap_ev=x["gap_ev"],doi=doi,doi_url=nz(x.get("source_url")),doi_type="article" if doi else "",
      source_group_id=x["source_group_id"],specimen_group_id=x["specimen_group_id"],source_locator=nz(x.get("source_locator")),
      sample_id=nz(x.get("source_record_id")),sample_conditions=nz(x.get("measurement_scope")),measurement_method=nz(x.get("measurement_method")),
      phase_label=nz(x.get("phase_label")),phase_purity=nz(x.get("phase_purity")),uncertainty_ev=nz(x.get("uncertainty_ev")),uncertainty_status=nz(x.get("uncertainty_status")),
      approximate=nz(x.get("approximate")),space_group=sg,structure_relation=nz(x.get("sample_cif_relation")),reference_structure_source=nz(e.get("structure_source"))))
out="Supplementary_Data_1_GapScope_experimental_bandgaps.csv"
with open(out,"w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows+new)
print(len(rows),len(new),len({r["composition_key"] for r in new}),len({r["source_group_id"] for r in new}),"with SG",sum(bool(r["space_group"]) for r in new),"with structure source",sum(bool(r["reference_structure_source"]) for r in new))
