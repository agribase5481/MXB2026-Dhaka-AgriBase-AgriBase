# URL of the article page
SDURL="https://www.sciencedirect.com/science/article/abs/pii/S2468784725001163?via=ihub"

# Step 1: Fetch the HTML and extract the PDF URL
curl -s -L -c cookiejar -b cookiejar "$SDURL" \
  | grep -oP 'pdfurl="[^"]+"' \
  | sed -e 's/pdfurl="//' -e 's/"//' \
  > pdfurl.txt

# Step 2: Download the PDF using the extracted URL
PDFURL=$(cat pdfurl.txt)
curl -L -c cookiejar -b cookiejar "https://www.sciencedirect.com${PDFURL}" -o article.pdf
