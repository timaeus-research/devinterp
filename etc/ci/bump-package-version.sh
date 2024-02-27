#!/usr/bin/env bash

set -e

TOML_PATH="pyproject.toml"
quote="'"
dquote='"'

VERSION_LINE="$(grep 'version\s*=\s*' "${TOML_PATH}")"
FULL_VERSION_NUMBER="$(echo "${VERSION_LINE}" | sed 's/version\s*=\s*//g; s/"//g; s/\s//g')"
VERSION_NUMBER="${FULL_VERSION_NUMBER%%+*}"
if [ -z "$1" ]; then
    # https://stackoverflow.com/a/4486087/377022
    NEW_VERSION_NUMBER="$(awk -F. '/[0-9]+\./{$NF++;print}' OFS=. <<< "${VERSION_NUMBER}")"
elif [[ "$1" == .* ]] || [[ "$1" == +* ]]; then
    NEW_VERSION_NUMBER="${VERSION_NUMBER}$1"
else
    NEW_VERSION_NUMBER="${VERSION_NUMBER}+$1"
fi
NEW_VERSION_LINE="$(echo "${VERSION_LINE}" | sed "s/${FULL_VERSION_NUMBER}/${NEW_VERSION_NUMBER}/g")"
echo "Updating ${TOML_PATH} from version ${FULL_VERSION_NUMBER} to ${NEW_VERSION_NUMBER}"
sed "s/${VERSION_LINE}/${NEW_VERSION_LINE}/g" -i "${TOML_PATH}"

# sanity check
AGAIN_VERSION_LINE="$(grep 'version\s*=\s*' "${TOML_PATH}")"
AGAIN_VERSION_NUMBER="$(echo "${AGAIN_VERSION_LINE}" | sed 's/version\s*=\s*//g; s/"//g; s/\s//g')"
if [ "${NEW_VERSION_NUMBER}" != "${AGAIN_VERSION_NUMBER}" ]; then
    echo "ERROR: Tried to change '${FULL_VERSION_NUMBER}' to '${NEW_VERSION_NUMBER}' in ${TOML_PATH},"
    echo "  but somehow ended up with '${AGAIN_VERSION_NUMBER}'."
    exit 1
fi
