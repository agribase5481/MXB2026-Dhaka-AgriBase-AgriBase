export SDURL="https://www.sciencedirect.com/science/article/abs/pii/S2468784725001163?via%3Dihub"
curl -Lc cookiejar "${SDURL}" | grep pdfurl | perl -pe 's|.* pdfurl=\"(.*?)\".*|\1|' > pdfurl
curl -Lc cookiejar "$(cat pdfurl)" > article.pdf
