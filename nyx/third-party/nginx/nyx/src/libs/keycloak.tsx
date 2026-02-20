
import Keycloak from "keycloak-js";


const keycloak = new Keycloak({
    url: "http://localhost:8080",
    realm: "assistant",
    clientId: "frontend"
});


export function getTokenMinValidity():number
{
    const exp = keycloak.tokenParsed?.exp;
    const iat = keycloak.tokenParsed?.iat;

    if (!exp || !iat) return 10;

    return Math.max(Math.floor((exp - iat) * 0.1), 5);
}

export default keycloak;
