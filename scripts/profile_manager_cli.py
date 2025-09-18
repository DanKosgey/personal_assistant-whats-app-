#!/usr/bin/env python3
"""
Profile Manager CLI - A command-line interface to manage user profiles in the database
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.services.profile_service import ProfileService
from server.models.profiles import UpsertProfileRequest, ProfileCreate
from server.db.profile_db import get_profile_db
from server.repositories.profile_repository import ProfileRepository, QueryOptions

class ProfileManagerCLI:
    def __init__(self):
        self.profile_service = ProfileService()
        self.repository = ProfileRepository()
        
    async def run(self):
        """Main CLI loop"""
        print("=== Profile Manager CLI ===")
        print("Managing profiles for Boniface's WhatsApp Assistant")
        print()
        
        while True:
            print("\nOptions:")
            print("1. List all profiles")
            print("2. View a specific profile")
            print("3. Add a new profile")
            print("4. Edit an existing profile")
            print("5. Delete a profile")
            print("6. Search profiles")
            print("7. Delete all profiles")
            print("8. Export profiles")
            print("9. Import profiles")
            print("10. Show statistics")
            print("11. Exit")
            
            choice = input("\nEnter your choice (1-11): ").strip()
            
            try:
                if choice == "1":
                    await self.list_profiles()
                elif choice == "2":
                    await self.view_profile()
                elif choice == "3":
                    await self.add_profile()
                elif choice == "4":
                    await self.edit_profile()
                elif choice == "5":
                    await self.delete_profile()
                elif choice == "6":
                    await self.search_profiles()
                elif choice == "7":
                    await self.delete_all_profiles()
                elif choice == "8":
                    await self.export_profiles()
                elif choice == "9":
                    await self.import_profiles()
                elif choice == "10":
                    await self.show_statistics()
                elif choice == "11":
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice. Please enter a number between 1 and 11.")
            except Exception as e:
                print(f"Error: {e}")
    
    async def list_profiles(self):
        """List all profiles in the database"""
        print("\n=== All Profiles ===")
        try:
            # Get all profiles from the repository
            options = QueryOptions(limit=1000)  # Get up to 1000 profiles
            profiles, total = await self.repository.list_profiles(options)
            
            if not profiles:
                print("No profiles found in the database.")
                return
                
            print(f"Found {total} profile(s):")
            print("-" * 80)
            
            for profile in profiles:
                name = getattr(profile, 'name', 'Unknown')
                phone = getattr(profile, 'phone', 'Unknown')
                last_seen = getattr(profile, 'last_seen', None)
                last_seen_str = last_seen.strftime("%Y-%m-%d %H:%M:%S") if last_seen else "Never"
                
                print(f"Phone: {phone}")
                print(f"  Name: {name}")
                print(f"  Last Seen: {last_seen_str}")
                print(f"  Tags: {getattr(profile, 'tags', [])}")
                print()
                
        except Exception as e:
            print(f"Error listing profiles: {e}")
    
    async def view_profile(self):
        """View a specific profile by phone number"""
        print("\n=== View Profile ===")
        phone = input("Enter phone number: ").strip()
        
        if not phone:
            print("Phone number is required.")
            return
            
        try:
            profile = await self.profile_service.get_or_create_profile(phone, auto_create=False)
            
            if not profile:
                print(f"No profile found for phone number: {phone}")
                return
                
            print(f"\n=== Profile Details for {phone} ===")
            print(f"ID: {getattr(profile, 'id', 'N/A')}")
            print(f"Name: {getattr(profile, 'name', 'N/A')}")
            print(f"Display Name: {getattr(profile, 'display_name', 'N/A')}")
            print(f"Language: {getattr(profile, 'language', 'N/A')}")
            print(f"Timezone: {getattr(profile, 'timezone', 'N/A')}")
            print(f"Persona: {getattr(profile, 'persona', 'N/A')}")
            print(f"Description: {getattr(profile, 'description', 'N/A')}")
            print(f"Consent: {getattr(profile, 'consent', False)}")
            print(f"Consent Date: {getattr(profile, 'consent_date', 'N/A')}")
            print(f"Last Seen: {getattr(profile, 'last_seen', 'N/A')}")
            print(f"Created At: {getattr(profile, 'created_at', 'N/A')}")
            print(f"Updated At: {getattr(profile, 'updated_at', 'N/A')}")
            print(f"Tags: {getattr(profile, 'tags', [])}")
            print(f"Attributes: {getattr(profile, 'attributes', {})}")
            
        except Exception as e:
            print(f"Error viewing profile: {e}")
    
    async def add_profile(self):
        """Add a new profile"""
        print("\n=== Add New Profile ===")
        phone = input("Enter phone number: ").strip()
        
        if not phone:
            print("Phone number is required.")
            return
            
        # Check if profile already exists
        existing_profile = await self.profile_service.get_or_create_profile(phone, auto_create=False)
        if existing_profile:
            print(f"Profile already exists for phone number: {phone}")
            return
            
        print("Enter profile details (leave blank for default values):")
        name = input("Name: ").strip() or None
        display_name = input("Display Name: ").strip() or None
        language = input("Language (default: en): ").strip() or "en"
        timezone = input("Timezone: ").strip() or None
        persona = input("Persona: ").strip() or None
        description = input("Description: ").strip() or None
        
        try:
            # Create profile data
            profile_data = ProfileCreate(
                phone=phone,
                name=name,
                display_name=display_name,
                language=language,
                timezone=timezone,
                persona=persona,
                description=description,
                consent=False,
                consent_date=None,
                last_seen=datetime.utcnow(),
                attributes={},
                tags=[],
                updated_by="cli_manager"
            )
            
            # Create the profile
            profile = await self.repository.create_profile(profile_data, "cli_manager")
            
            if profile:
                print(f"Successfully created profile for {phone}")
                print(f"Profile ID: {getattr(profile, 'id', 'N/A')}")
            else:
                print("Failed to create profile.")
                
        except Exception as e:
            print(f"Error creating profile: {e}")
    
    async def edit_profile(self):
        """Edit an existing profile"""
        print("\n=== Edit Profile ===")
        phone = input("Enter phone number: ").strip()
        
        if not phone:
            print("Phone number is required.")
            return
            
        try:
            # Get existing profile
            profile = await self.profile_service.get_or_create_profile(phone, auto_create=False)
            
            if not profile:
                print(f"No profile found for phone number: {phone}")
                return
                
            print(f"\nCurrent profile details for {phone}:")
            current_name = getattr(profile, 'name', None)
            current_display_name = getattr(profile, 'display_name', None)
            current_language = getattr(profile, 'language', 'en')
            current_timezone = getattr(profile, 'timezone', None)
            current_persona = getattr(profile, 'persona', None)
            current_description = getattr(profile, 'description', None)
            current_tags = getattr(profile, 'tags', [])
            current_attributes = getattr(profile, 'attributes', {})
            
            print(f"Current Name: {current_name or 'Not set'}")
            print(f"Current Display Name: {current_display_name or 'Not set'}")
            print(f"Current Language: {current_language}")
            print(f"Current Timezone: {current_timezone or 'Not set'}")
            print(f"Current Persona: {current_persona or 'Not set'}")
            print(f"Current Description: {current_description or 'Not set'}")
            print(f"Current Tags: {current_tags}")
            print(f"Current Attributes: {current_attributes}")
            
            # Ask for updates
            print("\nEnter new values (leave blank to keep current value):")
            new_name = input("Name: ").strip()
            new_display_name = input("Display Name: ").strip()
            new_language = input(f"Language (current: {current_language}): ").strip()
            new_timezone = input("Timezone: ").strip()
            new_persona = input("Persona: ").strip()
            new_description = input("Description: ").strip()
            
            # Prepare update fields
            update_fields = {}
            
            if new_name:
                update_fields['name'] = new_name if new_name.lower() != 'none' else None
            if new_display_name:
                update_fields['display_name'] = new_display_name if new_display_name.lower() != 'none' else None
            if new_language:
                update_fields['language'] = new_language
            if new_timezone:
                update_fields['timezone'] = new_timezone if new_timezone.lower() != 'none' else None
            if new_persona:
                update_fields['persona'] = new_persona if new_persona.lower() != 'none' else None
            if new_description:
                update_fields['description'] = new_description if new_description.lower() != 'none' else None
                
            # Handle tags
            tags_input = input("Tags (comma-separated, or 'clear' to remove all): ").strip()
            if tags_input.lower() == 'clear':
                update_fields['tags'] = []
            elif tags_input:
                update_fields['tags'] = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
                
            # Handle attributes
            print("For attributes, you can:")
            print("1. Add/Update an attribute")
            print("2. Remove an attribute")
            print("3. Skip attribute changes")
            attr_choice = input("Choose option (1-3): ").strip()
            
            if attr_choice == "1":
                attr_key = input("Attribute key: ").strip()
                attr_value = input("Attribute value: ").strip()
                if attr_key and attr_value:
                    current_attributes[attr_key] = attr_value
                    update_fields['attributes'] = current_attributes
            elif attr_choice == "2":
                if current_attributes:
                    print("Current attributes:")
                    for key, value in current_attributes.items():
                        print(f"  {key}: {value}")
                    attr_key = input("Attribute key to remove: ").strip()
                    if attr_key in current_attributes:
                        del current_attributes[attr_key]
                        update_fields['attributes'] = current_attributes
                    else:
                        print("Attribute key not found.")
                else:
                    print("No attributes to remove.")
            
            if not update_fields:
                print("No changes to make.")
                return
                
            # Create update request
            update_request = UpsertProfileRequest(
                phone=phone,
                fields=update_fields,
                reason="Updated via CLI manager",
                expected_version=None,
                session_id=None
            )
            
            # Update the profile
            response = await self.profile_service.upsert_profile(update_request, actor="cli_manager")
            
            if response.success:
                print("Profile updated successfully!")
                print(f"Updated profile ID: {getattr(response.profile, 'id', 'N/A')}")
            else:
                print(f"Failed to update profile: {response.message}")
                
        except Exception as e:
            print(f"Error editing profile: {e}")
    
    async def delete_profile(self):
        """Delete a profile"""
        print("\n=== Delete Profile ===")
        phone = input("Enter phone number: ").strip()
        
        if not phone:
            print("Phone number is required.")
            return
            
        try:
            # Get existing profile by phone number
            profile = await self.profile_service.get_or_create_profile(phone, auto_create=False)
            
            if not profile:
                print(f"No profile found for phone number: {phone}")
                return
                
            # Confirm deletion
            confirm = input(f"Are you sure you want to delete profile for {phone}? (yes/no): ").strip().lower()
            
            if confirm != "yes":
                print("Deletion cancelled.")
                return
                
            # Delete the profile using the service
            success = await self.profile_service.delete_profile(phone, "cli_manager", "Deleted via CLI manager")
            
            if success:
                print(f"Profile for {phone} deleted successfully!")
            else:
                print(f"Failed to delete profile for {phone}")
                
        except Exception as e:
            print(f"Error deleting profile: {e}")
    
    async def delete_all_profiles(self):
        """Delete all profiles from the database"""
        print("\n=== Delete All Profiles ===")
        
        # First, list the number of profiles to be deleted
        try:
            options = QueryOptions(limit=1000)
            profiles, total = await self.repository.list_profiles(options)
            
            if total == 0:
                print("No profiles found in the database.")
                return
                
            print(f"WARNING: This will delete ALL {total} profiles from the database.")
            print("This action cannot be undone.")
            
            confirm = input(f"\nType 'DELETE ALL {total} PROFILES' to confirm: ").strip()
            
            if confirm != f"DELETE ALL {total} PROFILES":
                print("Deletion cancelled.")
                return
                
            # Delete all profiles
            deleted_count = 0
            for profile in profiles:
                try:
                    success = await self.profile_service.delete_profile(profile.phone, "cli_manager", "Deleted via CLI manager - Delete All")
                    if success:
                        deleted_count += 1
                except Exception as e:
                    print(f"Error deleting profile {profile.id}: {e}")
                    
            print(f"Successfully deleted {deleted_count} out of {total} profiles.")
                
        except Exception as e:
            print(f"Error deleting all profiles: {e}")
    
    async def search_profiles(self):
        """Search profiles by name or phone number"""
        print("\n=== Search Profiles ===")
        search_term = input("Enter search term (name or phone): ").strip()
        
        if not search_term:
            print("Search term is required.")
            return
            
        try:
            # Search profiles using the repository
            print("Searching profiles...")
            
            # Use the repository's search method
            profiles = await self.repository.search_profiles(search_term)
            
            if not profiles:
                print(f"No profiles found matching '{search_term}'")
                return
                
            print(f"Found {len(profiles)} matching profile(s):")
            print("-" * 80)
            
            for profile in profiles:
                name = getattr(profile, 'name', 'Unknown')
                phone = getattr(profile, 'phone', 'Unknown')
                last_seen = getattr(profile, 'last_seen', None)
                last_seen_str = last_seen.strftime("%Y-%m-%d %H:%M:%S") if last_seen else "Never"
                
                print(f"Phone: {phone}")
                print(f"  Name: {name}")
                print(f"  Last Seen: {last_seen_str}")
                print()
                
        except Exception as e:
            print(f"Error searching profiles: {e}")
    
    async def export_profiles(self):
        """Export all profiles to a JSON file"""
        print("\n=== Export Profiles ===")
        
        try:
            # Get all profiles
            options = QueryOptions(limit=10000)  # Get up to 10000 profiles
            profiles, total = await self.repository.list_profiles(options)
            
            if not profiles:
                print("No profiles found to export.")
                return
            
            # Create export data
            export_data = []
            for profile in profiles:
                profile_dict = {
                    "id": str(getattr(profile, 'id', '')),
                    "phone": getattr(profile, 'phone', ''),
                    "name": getattr(profile, 'name', None),
                    "display_name": getattr(profile, 'display_name', None),
                    "language": getattr(profile, 'language', 'en'),
                    "timezone": getattr(profile, 'timezone', None),
                    "persona": getattr(profile, 'persona', None),
                    "description": getattr(profile, 'description', None),
                    "consent": getattr(profile, 'consent', False),
                    "consent_date": getattr(profile, 'consent_date', None),
                    "last_seen": getattr(profile, 'last_seen', None),
                    "created_at": getattr(profile, 'created_at', None),
                    "updated_at": getattr(profile, 'updated_at', None),
                    "tags": getattr(profile, 'tags', []),
                    "attributes": getattr(profile, 'attributes', {}),
                }
                export_data.append(profile_dict)
            
            # Write to file
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"profiles_export_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"Successfully exported {total} profiles to {filename}")
            
        except Exception as e:
            print(f"Error exporting profiles: {e}")
    
    async def import_profiles(self):
        """Import profiles from a JSON file"""
        print("\n=== Import Profiles ===")
        filename = input("Enter path to JSON file: ").strip()
        
        if not filename:
            print("File path is required.")
            return
            
        try:
            # Read the JSON file
            with open(filename, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            if not isinstance(import_data, list):
                print("Invalid file format. Expected a list of profiles.")
                return
            
            print(f"Found {len(import_data)} profiles in the file.")
            
            # Confirm import
            confirm = input("Do you want to import these profiles? (yes/no): ").strip().lower()
            if confirm != "yes":
                print("Import cancelled.")
                return
            
            # Import profiles
            imported_count = 0
            for profile_data in import_data:
                try:
                    # Check if profile already exists
                    phone = profile_data.get('phone')
                    if not phone:
                        print(f"Skipping profile with missing phone number")
                        continue
                        
                    existing_profile = await self.profile_service.get_or_create_profile(phone, auto_create=False)
                    
                    if existing_profile:
                        print(f"Profile for {phone} already exists, skipping...")
                        continue
                    
                    # Create new profile
                    profile_create = ProfileCreate(
                        phone=phone,
                        name=profile_data.get('name'),
                        display_name=profile_data.get('display_name'),
                        language=profile_data.get('language', 'en'),
                        timezone=profile_data.get('timezone'),
                        persona=profile_data.get('persona'),
                        description=profile_data.get('description'),
                        consent=profile_data.get('consent', False),
                        consent_date=profile_data.get('consent_date'),
                        last_seen=profile_data.get('last_seen'),
                        attributes=profile_data.get('attributes', {}),
                        tags=profile_data.get('tags', []),
                        updated_by="cli_manager_import"
                    )
                    
                    profile = await self.repository.create_profile(profile_create, "cli_manager_import")
                    if profile:
                        imported_count += 1
                        
                except Exception as e:
                    print(f"Error importing profile for {profile_data.get('phone', 'unknown')}: {e}")
            
            print(f"Successfully imported {imported_count} profiles.")
            
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except json.JSONDecodeError:
            print(f"Invalid JSON format in file: {filename}")
        except Exception as e:
            print(f"Error importing profiles: {e}")
    
    async def show_statistics(self):
        """Show profile statistics"""
        print("\n=== Profile Statistics ===")
        
        try:
            # Get profile statistics
            stats = await self.repository.get_profile_stats()
            
            print(f"Total Profiles: {stats.get('total_profiles', 0)}")
            print(f"Active Profiles: {stats.get('active_profiles', 0)}")
            print(f"Consented Profiles: {stats.get('consented_profiles', 0)}")
            print(f"Recent Interactions (24h): {stats.get('recent_interactions', 0)}")
            
            # Top personas
            top_personas = stats.get('top_personas', [])
            if top_personas:
                print("\nTop Personas:")
                for persona in top_personas[:5]:  # Show top 5
                    print(f"  {persona['persona']}: {persona['count']} profiles")
            
            # Top languages
            top_languages = stats.get('top_languages', [])
            if top_languages:
                print("\nTop Languages:")
                for language in top_languages[:5]:  # Show top 5
                    print(f"  {language['language']}: {language['count']} profiles")
                    
        except Exception as e:
            print(f"Error retrieving statistics: {e}")

async def main():
    """Main entry point"""
    # Initialize database connection
    try:
        # Initialize the profile database connection
        print("Initializing profile database connection...")
        profile_db = await get_profile_db()
        
        # Create and run the CLI
        cli = ProfileManagerCLI()
        await cli.run()
        
    except Exception as e:
        print(f"Failed to start Profile Manager CLI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())