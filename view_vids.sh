(cat videos/*json|jq 2>&1) | egrep '"title"|descr|"id"'|less
