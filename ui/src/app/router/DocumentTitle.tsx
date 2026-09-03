import { useEffect } from "react";
import { useLocation } from "react-router-dom";

import { documentTitleForLocation } from "@/app/router/routePresentation";

export function DocumentTitle() {
  const location = useLocation();

  useEffect(() => {
    document.title = documentTitleForLocation(
      location.pathname,
      location.search,
    );
  }, [location.pathname, location.search]);

  return null;
}
