import weaviate
import sys

try:
    # Connexion à Weaviate
    client = weaviate.connect_to_local(skip_init_checks=True)
    
    try:
        # Récupération de la collection
        if not client.collections.exists("test2"):
            print("La collection 'qwanza_docs' n'existe pas!")
            sys.exit(1)
            
        collection = client.collections.get("test2")
        
        # 1. Récupérer tous les objets pour compter
        results = collection.query.fetch_objects()
        total_objects = len(results.objects)
        print(f"\nNombre total d'objets dans la collection : {total_objects}")
        
        if total_objects > 0:
            # 2. Afficher les 5 premiers objets
            print("\nAperçu des 5 premiers objets :")
            for i, obj in enumerate(results.objects[:5]):
                print(f"\nObjet {i+1}:")
                content = obj.properties.get('content', '')
                print(f"Content: {content[:200]}...")  # Affiche les 200 premiers caractères
                source = obj.properties.get('source', 'Non spécifié')
                print(f"Source: {source}")
                print("-" * 80)
    
    finally:
        client.close()

except Exception as e:
    print(f"Erreur : {str(e)}")
    sys.exit(1)