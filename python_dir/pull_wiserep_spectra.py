from wiserep_api import download_target_spectra
import os
import shutil

"""
# Unique targets extracted from your file list
targets = [
    "PTF10zdk", "PTF10hml", "2009hi", "2012cp",
    "PTF09bce", "PTF09fim", "PTF09fqa", "PTF09tm", "PTF10vgv",
    "PTF10woa", "PTF11bpx", "PTF11dhf", "PTF11rnu", "PTF10bdr",
    "PTF11hzx", "PTF12ljq", "SN2018gea", "SN2018esq", "2017hbv",
    "2017dgj", "2017dgk", "2017dgy", "2017dio", "2017diz",
    "2017eef", "2017eje", "2017gaw", "2017gat", "2017ggl",
    "2017hkn", "2017hyn", "2017jfw", "2018akm", "2018ame",
    "2018aqn", "2018beh", "2018bui", "ASASSN-15oz", "SN2018fer",
    "SN2018giu", "SN2018gjx", "SN2019so", "ZTF19aamuvak",
    "SNLS-07D3hl", "SNLS-06D4dr", "SNLS-06D3cn", "SNLS-05D3ax",
    "SNLS-05D1if", "SNLS-05D3cx", "2005ky", "ZTF22aasxgjp",
    "ZTF18adbmrug", "ZTF18abvvmdf", "2019ouq", "2019mjo",
    "2019ccm", "2019yvq"
]

for sn in targets:
    try:
        download_target_spectra(sn, file_type='ascii')
        print(f"✓ {sn}")
    except Exception as e:
        print(f"✗ {sn}: {e}")
"""



# The specific files you want
target_files = [
    "PTF10zdk_2010-11-02_00-00-00_Mayall_RC-Spec_PTF.ascii",
    "PTF10hml_2010-05-21_00-00-00_P200_DBSP_PTF.ascii",
    "2009hi_2009-08-16_00-00-00_P200_DBSP_PTF.ascii",
    "2011bm_2455692.5000_Mayall_RC-Spec_PTF.ascii",
    "2012cp_2012-07-19_00-00-00_P200_DBSP_PTF.ascii",
    "PTF09bce_2009-07-25_09-56-00.680000_Lick-3m_KAST_PTF_0.ascii",
    "PTF09fim_2009-10-12_00-00-00_WHT-4.2m_ISIS_PTF.ascii",
    "PTF09fqa_2009-10-22_00-00-00.000000_Keck1_LRIS_PTF_0.ascii",
    "PTF09tm_2009-07-25_05-50-14_Lick-3m_KAST_PTF.ascii",
    "PTF10vgv_2455499.50_Keck1_LRIS_PTF.clean",
    "PTF10woa_2010-11-04_00-00-00_Mayall_RC-Spec_PTF.ascii",
    "PTF11bpx_2011-05-25_00-00-00.000000_P200_DBSP_PTF.ascii",
    "PTF11dhf_2011-05-25_00-00-00.000000_P200_DBSP_PTF.ascii",
    "PTF11rnu_2012-01-27_00-00-00_Mayall_RC-Spec_PTF.ascii",
    "PTF10bdr_2010-02-11_08-55-29_Keck1_LRIS_PTF.ascii",
    "PTF11hzx_2011-08-28_07-12-00_P200_DBSP_PTF.ascii",
    "PTF12ljq_2012-12-05_00-00-00_P200_DBSP_PTF.flm",
    "SN2018gea_2018-11-15_02-35-59.570_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "SN2018esq_2018-08-11_05-02-53.967_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017hbv_2017-10-12_08-22-38_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017dgj_2017-04-23_06-24-43_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017dgk_2017-04-24_02-29-45_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017dgy_2017-04-29_03-49-31_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017dio_2017-04-30_02-11-39_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017diz_2017-04-29_23-07-09_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017eef_2017-06-03_03-20-42_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017eje_2017-06-02_05-30-53_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017gaw_2017-08-17_08-17-20_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017gat_2017-08-17_04-07-55_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017ggl_2017-08-29_07-09-41_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017hkn_2017-10-29_00-52-22_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017hyn_2017-11-12_05-59-19_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2017jfw_2017-12-30_08-25-14_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2018akm_2018-03-28_09-23-08_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2018ame_2018-03-27_08-02-46_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2018aqn_2018-04-08_08-09-43_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2018beh_2018-05-06_00-58-49_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "2018bui_2018-05-23_02-43-28_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "ASASSN-15oz_2015-11-19_00-40-45_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "SN2018fer_2018-09-09_03-47-36.289_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "SN2018giu_2018-09-17_03-41-09.814_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "SN2018gjx_2018-10-03_06-51-59.767_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "SN2019so_2019-01-15_05-16-38.888_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "ZTF19aamuvak_2019-03-19_07-17-22.978_ESO-NTT_EFOSC2-NTT_ePESSTO.asci",
    "SNLS-07D3hl_2007-06-09_00-00-00_Gemini-N_GMOS_SNLS",
    "SNLS-06D4dr_2006-08-19_00-00-00_Gemini-N_GMOS_SNLS",
    "SNLS-06D3cn_2006-03-06_00-00-00_Gemini-N_GMOS_SNLS.asc",
    "SNLS-05D3ax_2005-02-11_00-00-00_Gemini-N_GMOS_SNLS.asc",
    "SNLS-05D1if_2005-12-01_00-00-00_Keck1_LRIS_SNLS",
    "SNLS-05D3ax_2005-02-11_00-00-00_Gemini-N_GMOS_SNLS.asc",
    "SNLS-05D3cx_2005-04-09_00-00-00_Gemini-N_GMOS_SNLS.asc",
    "SNLS-06D3cn_2006-03-06_00-00-00_Gemini-N_GMOS_SNLS.asc",
    "2005ky_2005-11-30_00-00-00_Keck1_LRIS_SNLS",
    "ZTF22aasxgjp_2022-09-01_03-24-43_P60_SEDM_ZTF.ascii",
    "ZTF18adbmrug_2019-01-01_22-47-00_WHT-4.2m_ISIS_ZTF.ascii",
    "ZTF18abvvmdf_2458376.619_P60_SEDM_ZTF.ascii",
    "2019ouq_2019-08-04_00-00-00_P200_DBSP_ZTF",
    "2019mjo_2019-08-01_00-00-00_P200_DBSP_ZTF",
    "2019ccm_2019-04-06_00-00-00_Keck1_LRIS_ZTF",
    "2019yvq_2020-01-24_14-44-38_Keck1_LRIS_ZTF.ascii",
]

spectra_dir = "spectra"
output_dir = "Dwayne"
os.makedirs(output_dir, exist_ok=True)

# Build a lookup: filename -> full path, by walking the spectra directory
file_index = {}
for root, dirs, files in os.walk(spectra_dir):
    for fname in files:
        file_index[fname] = os.path.join(root, fname)

# Find and copy each target file
found, missing = [], []
for target in set(target_files):  # set() handles the duplicate SNLS entries
    if target in file_index:
        shutil.copy2(file_index[target], os.path.join(output_dir, target))
        found.append(target)
    else:
        missing.append(target)

print(f"\n✓ Copied {len(found)} files to '{output_dir}/'")
if missing:
    print(f"✗ {len(missing)} files not found:")
    for f in sorted(missing):
        print(f"   {f}")