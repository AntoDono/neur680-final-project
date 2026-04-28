#!/bin/bash
#
#================================================================
# aseg_lh_rh_download_oasis_freesurfer.sh
#================================================================
#
# Usage: ./aseg_lh_rh_download_oasis_freesurfer.sh <input_file.csv> <directory_name> <nitrc_ir_username>
#
# Fetches Freesurfer XML from NITRC-IR for each subject and parses:
#   - aseg volumes   -> aseg_volumes.csv
#   - lh aparc stats -> lh_thickness.csv
#   - rh aparc stats -> rh_thickness.csv
#
# Required inputs:
# <input_file.csv>    - Unix formatted CSV with a freesurfer_id column
#                       (e.g. OAS30001_Freesurfer53_d0129)
# <directory_name>    - Directory path to save output CSVs to
# <nitrc_ir_username> - Your NITRC IR username (prompted for password)
#
# Output:
#   directory_name/aseg_volumes.csv
#   directory_name/lh_thickness.csv
#   directory_name/rh_thickness.csv
#
# Requires: curl, python3
#================================================================

unset module

startSession() {
    local COOKIE_JAR=.cookies-$(date +%Y%M%d%s).txt
    curl -k -s -u ${USERNAME}:${PASSWORD} --cookie-jar ${COOKIE_JAR} "https://www.nitrc.org/ir/data/JSESSION" > /dev/null
    echo ${COOKIE_JAR}
}

escape_chars_for_URL() {
    local input=${1}
    output=`echo "${input}" | sed -e 's/%/%25/g;' | sed -e 's/ /%20/g; s/</%3C/g; s/>/%3E/g; s/#/%23/g; s/+/%2B/g; s/{/%7B/g; s/}/%7D/g; s/|/%7C/g; s/\\\/%5C/g; s/\^/%5E/g; s/~/%7E/g; s/\[/%5B/g; s/\]/%5D/g; s/\`/%60/g; s/;/%3B/g; s/\//%2F/g; s/?/%3F/g; s/:/%3A/g; s/@/%40/g; s/=/%3D/g; s/&/%26/g; s/\\$/%24/g'`
    echo ${output}
}

get() {
    local URL=${1}
    curl -H 'Expect:' --keepalive-time 2 -k --cookie ${COOKIE_JAR} "${URL}"
}

endSession() {
    curl -i -k --cookie ${COOKIE_JAR} -X DELETE "https://www.nitrc.org/ir/data/JSESSION"
    rm -f ${COOKIE_JAR}
}

if [ ${#@} == 0 ]; then
    echo ""
    echo "OASIS Freesurfer XML parser - aseg + lh/rh aparc"
    echo ""
    echo "Usage: $0 input_file.csv directory_name nitrc_username"
    echo ""
    echo "<input_file.csv>: Unix formatted CSV with freesurfer_id column"
    echo "    e.g. OAS30001_Freesurfer53_d0129"
    echo "<directory_name>: Directory to save output CSVs"
    echo "<nitrc_ir_username>: Your NITRC IR username"
    echo ""
    echo "Output:"
    echo "    directory_name/aseg_volumes.csv"
    echo "    directory_name/lh_thickness.csv"
    echo "    directory_name/rh_thickness.csv"
    echo ""
else
    INFILE=$1
    DIRNAME=$2
    USERNAME=$3

    if [ ! -d $DIRNAME ]; then
        mkdir -p $DIRNAME
    fi

    read -s -p "Enter your password for accessing OASIS data on NITRC IR: " PASSWORD
    echo ""

    USERNAME=`escape_chars_for_URL "${USERNAME}"`
    PASSWORD=`escape_chars_for_URL "${PASSWORD}"`

    COOKIE_JAR=$(startSession)

    # Temp directory for raw XMLs
    TMPDIR=${DIRNAME}/tmp_xml
    mkdir -p ${TMPDIR}

    echo ""
    echo "Fetching Freesurfer XMLs..."
    echo ""

    sed 1d $INFILE | while IFS=, read -r FREESURFER_ID; do

        FREESURFER_ID=`echo $FREESURFER_ID | tr -d '\r'`

        if [ -z "$FREESURFER_ID" ]; then
            continue
        fi

        SUBJECT_ID=`echo $FREESURFER_ID | cut -d_ -f1`
        DAYS_FROM_ENTRY=`echo $FREESURFER_ID | cut -d_ -f3`
        EXPERIMENT_LABEL=${SUBJECT_ID}_MR_${DAYS_FROM_ENTRY}

        PROJECT_ID=OASIS3
        if [[ "${SUBJECT_ID}" == "OAS4"* ]]; then
            PROJECT_ID=OASIS4
        fi

        XML_URL="https://www.nitrc.org/ir/data/archive/projects/${PROJECT_ID}/subjects/${SUBJECT_ID}/experiments/${EXPERIMENT_LABEL}/assessors/${FREESURFER_ID}?format=xml"
        XML_FILE="${TMPDIR}/${FREESURFER_ID}.xml"

        echo "Fetching XML for ${FREESURFER_ID} ..."
        get "${XML_URL}" > "${XML_FILE}"

        if grep -q "fs:Freesurfer" "${XML_FILE}"; then
            echo "  ✓ Got XML for ${FREESURFER_ID}"
        else
            echo "  ✗ Failed or empty XML for ${FREESURFER_ID}"
            rm -f "${XML_FILE}"
        fi

    done < $INFILE

    endSession

    echo ""
    echo "Parsing XMLs into CSVs..."
    echo ""

    python3 - "${TMPDIR}" "${DIRNAME}" << 'PYEOF'
import os
import sys
import xml.etree.ElementTree as ET
import csv

tmpdir = sys.argv[1]
outdir = sys.argv[2]

NS = {
    'fs': 'http://nrg.wustl.edu/fs',
    'xnat': 'http://nrg.wustl.edu/xnat'
}

aseg_rows = []
lh_rows = []
rh_rows = []

xml_files = sorted([f for f in os.listdir(tmpdir) if f.endswith('.xml')])
print(f"Parsing {len(xml_files)} XML files...")

for fname in xml_files:
    fpath = os.path.join(tmpdir, fname)
    freesurfer_id = fname.replace('.xml', '')
    subject_id = freesurfer_id.split('_')[0]

    try:
        tree = ET.parse(fpath)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  Skipping {fname}: XML parse error - {e}")
        continue

    # ---------- ASEG VOLUMES ----------
    vol = root.find('.//fs:volumetric', NS)
    if vol is not None:
        aseg_row = {'subject_id': subject_id, 'freesurfer_id': freesurfer_id}

        global_fields = [
            'ICV', 'lhCortexVol', 'rhCortexVol', 'CortexVol',
            'SubCortGrayVol', 'TotalGrayVol', 'SupraTentorialVol',
            'lhCorticalWhiteMatterVol', 'rhCorticalWhiteMatterVol',
            'CorticalWhiteMatterVol', 'BrainSegVol', 'BrainSegVolNotVent',
            'BrainSegVolNotVentSurf', 'SupraTentorialVolNotVent',
            'MaskVol', 'lhSurfaceHoles', 'rhSurfaceHoles', 'SurfaceHoles'
        ]
        for field in global_fields:
            el = vol.find(f'fs:{field}', NS)
            aseg_row[field] = el.text.strip() if el is not None else ''

        for region in vol.findall('.//fs:region', NS):
            name = region.get('name', '').replace('-', '_').replace(' ', '_')
            vol_el = region.find('fs:Volume', NS)
            aseg_row[f'vol_{name}'] = vol_el.text.strip() if vol_el is not None else ''

        aseg_rows.append(aseg_row)

    # ---------- LH / RH APARC ----------
    for hemi_el in root.findall('.//fs:hemisphere', NS):
        hemi = hemi_el.get('name')
        hemi_row = {'subject_id': subject_id, 'freesurfer_id': freesurfer_id}

        for region in hemi_el.findall('.//fs:region', NS):
            rname = region.get('name', '')
            for measure in ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']:
                el = region.find(f'fs:{measure}', NS)
                hemi_row[f'{rname}_{measure}'] = el.text.strip() if el is not None else ''

        if hemi == 'left':
            lh_rows.append(hemi_row)
        elif hemi == 'right':
            rh_rows.append(hemi_row)

    print(f"  Parsed {freesurfer_id}")

def write_csv(rows, filepath):
    if not rows:
        print(f"  No data to write for {filepath}")
        return
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in keys})
    print(f"  Wrote {len(rows)} rows -> {filepath}")

write_csv(aseg_rows, os.path.join(outdir, 'aseg_volumes.csv'))
write_csv(lh_rows,   os.path.join(outdir, 'lh_thickness.csv'))
write_csv(rh_rows,   os.path.join(outdir, 'rh_thickness.csv'))

print("\nDone!")
PYEOF

    echo ""
    echo "Cleaning up temp XMLs..."
    rm -rf ${TMPDIR}
    echo ""
    echo "All done! Output CSVs saved to: ${DIRNAME}"
    echo "  ${DIRNAME}/aseg_volumes.csv"
    echo "  ${DIRNAME}/lh_thickness.csv"
    echo "  ${DIRNAME}/rh_thickness.csv"

fi