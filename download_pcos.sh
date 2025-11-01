export SDURL="http://www.sciencedirect.com/science/article/pii/S0169433215012131"
curl -Lc cookiejar "${SDURL}" | grep pdfurl | perl -pe 's|.* pdfurl=\"(.*?)\".*|\1|' > pdfurl
curl -Lc cookiejar "$(cat pdfurl)" > article.pdf
